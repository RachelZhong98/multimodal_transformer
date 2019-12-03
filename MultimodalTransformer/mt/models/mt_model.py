import torch.nn as nn
from torchtext.vocab import GloVe
from allennlp.nn.util import add_sentence_boundary_token_ids, sequence_cross_entropy_with_logits
from allennlp.data import Vocabulary
from mt.config import Config
import functools
from typing import Dict, List, Tuple, Optional
import torch
from allennlp.nn.beam_search import BeamSearch
from mt.modules import UpDownCell
from mt.utils.decoding import select_best_beam
class MultimodelTransformer(nn.Module):
    def __init__(self, vocabulary: Vocabulary, image_feature_size: int,num_attention_block: int,
                    caption_length: int, vocab_size: int, embedding_size: int, hidden_size: int,
                    feed_forward_dim: int, encode_dim: int, num_heads: int, attention_projection_size: int,
                    max_caption_length: int, beam_size:int):
        super().__init__()
        self.image_feature_size = image_feature_size
        self.num_attention_block = num_attention_block
        self.caption_length = caption_length
        self.vocab_size = vocabulary.get_vocab_size()
        self._vocabulary = vocabulary
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.feed_forward_dim = feed_forward_dim
        self.encode_dim = encode_dim
        self.num_heads = num_heads
        self.max_caption_length = 16

        self._pad_index = vocabulary.get_token_index("@@UNKNOWN@@")
        self._boundary_index = vocabulary.get_token_index("@@BOUNDARY@@")
        self.attention_projection_size = attention_projection_size

        self.adapt_image_features = nn.Linear(self.image_feature_size, self.feed_forward_dim)
        self.adapt_again = nn.Linear(self.feed_forward_dim, self.image_feature_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feed_forward_dim, nhead=self.num_heads, dim_feedforward=self.feed_forward_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attention_block)

        glove_vectors = self._initialize_glove()
        self._embedding_layer = nn.Embedding.from_pretrained(
                glove_vectors, freeze=True, padding_idx=self._pad_index
            )
        self._updown_cell = UpDownCell(
            self.image_feature_size, self.embedding_size, self.hidden_size, attention_projection_size
        )
        self._output_projection = nn.Sequential(
                nn.Linear(hidden_size, self.embedding_size), nn.Tanh()
            )
        self._output_layer = nn.Linear(self.embedding_size, self.vocab_size, bias=False)
        self._output_layer.weight = self._embedding_layer.weight
        self._log_softmax = nn.LogSoftmax(dim=1)
        
        # self.tgt_mask = None
        # # self.caption_lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=hidden_size, num_layers=1)
        # self.caption_linear = nn.Linear(self.embedding_size, self.hidden_size)
        # decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, nhead=self.num_heads, dim_feedforward=self.feed_forward_dim)
        # self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = self.num_attention_block)


        # #check input dim 
        # self.final = nn.Linear(self.caption_length, self.vocab_size)

        BeamSearchClass = BeamSearch
        self._beam_search = BeamSearchClass(
            self._boundary_index,
            max_steps=max_caption_length,
            beam_size=beam_size,
            per_node_beam_size=beam_size // 2,
        )


    @classmethod
    def from_config(cls, config: Config, **kwargs):
        r"""Instantiate this class directly from a :class:`~updown.config.Config`."""
        _C = config

        vocabulary = kwargs.pop("vocabulary")
        return cls(
            vocabulary=vocabulary,
            image_feature_size=_C.MODEL.IMAGE_FEATURE_SIZE,
            num_attention_block= _C.MODEL.NUM_ATTENTION_BLOCK,
            caption_length = _C.MODEL.CAPTION_LENGTH,
            vocab_size = _C.MODEL.VOCAB_SIZE,
            embedding_size=_C.MODEL.EMBEDDING_SIZE,
            hidden_size=_C.MODEL.HIDDEN_SIZE,
            feed_forward_dim= _C.MODEL.FEED_FORWARD_DIM,
            encode_dim = _C.MODEL.ENCODE_DIM,
            num_heads = _C.MODEL.NUM_HEADS,
            attention_projection_size = _C.MODEL.ATTENTION_PROJECTION_SIZE,
            max_caption_length = _C.DATA.MAX_CAPTION_LENGTH,
            beam_size = _C.MODEL.BEAM_SIZE
        )

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    # def encode(self, processed_image_features):
    #     return self.encoder(processed_image_features)
 
    # def decode(self, embedded_words):
    #     return self.decoder(embedded_words)

    # def embed(self, tokenized_words):
    #     return self.caption_lstm(tokenized_words)

    def forward(self, image_features, 
        caption_tokens:Optional[torch.Tensor] = None,
        device:Optional[int]=0):
        states = None
        batch_size, num_boxes, image_feature_size = image_features.size()
        if self.training and caption_tokens is not None:
            # Add "@@BOUNDARY@@" tokens to caption sequences.
            caption_tokens, _ = add_sentence_boundary_token_ids(
                caption_tokens,
                (caption_tokens != self._pad_index),
                self._boundary_index,
                self._boundary_index,
            )
            
            batch_size, max_caption_length = caption_tokens.size()

            # shape: (batch_size, max_caption_length)
            tokens_mask = caption_tokens != self._pad_index

            # The last input from the target is either padding or the boundary token.
            # Either way, we don't have to process it.
            num_decoding_steps = max_caption_length - 1

            image_features = self.adapt_image_features(image_features)
            image_features = self.encoder(image_features)
            image_features = self.adapt_again(image_features)

            step_logits: List[torch.Tensor] = []
            for timestep in range(num_decoding_steps):
                # shape: (batch_size,)
                input_tokens = caption_tokens[:, timestep]
                
                
                # shape: (batch_size, num_classes)
                output_logits, states = self._decode_step(image_features, input_tokens, states)

                # list of tensors, shape: (batch_size, 1, vocab_size)
                step_logits.append(output_logits.unsqueeze(1))

            # shape: (batch_size, num_decoding_steps)
            logits = torch.cat(step_logits, 1)

            # Skip first time-step from targets for calculating loss.
            output_dict = {
                "loss": self._get_loss(
                    logits, caption_tokens[:, 1:].contiguous(), tokens_mask[:, 1:].contiguous()
                )
            }
            
        else:
            num_decoding_steps = self.max_caption_length

            image_features = self.adapt_image_features(image_features)
            image_features = self.encoder(image_features)
            image_features = self.adapt_again(image_features)

            start_predictions = image_features.new_full((batch_size,), self._boundary_index).long()
    
            # Add image features as a default argument to match callable signature acceptable by
            # beam search class (previous predictions and states only).
            beam_decode_step = functools.partial(self._decode_step, image_features)

            # shape (all_top_k_predictions): (batch_size, net_beam_size, num_decoding_steps)
            # shape (log_probabilities): (batch_size, net_beam_size)
            # if self._use_cbs:
            #     all_top_k_predictions, log_probabilities = self._beam_search.search(
            #         start_predictions, states, beam_decode_step, fsm
            #     )
            #     best_beam = select_best_beam_with_constraints(
            #         all_top_k_predictions,
            #         log_probabilities,
            #         num_constraints,
            #         self._min_constraints_to_satisfy,
            #     )
            # else:
            
            all_top_k_predictions, log_probabilities = self._beam_search.search(
                start_predictions, states, beam_decode_step
            )
            best_beam = select_best_beam(all_top_k_predictions, log_probabilities)

            # shape: (batch_size, num_decoding_steps)
            output_dict = {"predictions": best_beam}
            
        
        

        return output_dict
        
        

        

    def _initialize_glove(self) -> torch.Tensor:
        glove = GloVe(name="42B", dim=300)###
        glove_vectors = torch.zeros(self._vocabulary.get_vocab_size(), 300)

        for word, i in self._vocabulary.get_token_to_index_vocabulary().items():
            if word in glove.stoi:
                glove_vectors[i] = glove.vectors[glove.stoi[word]]
            # #Added
            elif word != self._pad_index:
                # Initialize by random vector.
                glove_vectors[i] = 2 * torch.randn(300) - 1

        return glove_vectors

    

    
    def _get_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute cross entropy loss of predicted caption (logits) w.r.t. target caption. The cross
        entropy loss of caption is cross entropy loss at each time-step, summed.
        Parameters
        ----------
        logits: torch.Tensor
            A tensor of shape ``(batch_size, max_caption_length - 1, vocab_size)`` containing
            unnormalized log-probabilities of predicted captions.
        targets: torch.Tensor
            A tensor of shape ``(batch_size, max_caption_length - 1)`` of tokenized target
            captions.
        target_mask: torch.Tensor
            A mask over target captions, elements where mask is zero are ignored from loss
            computation. Here, we ignore ``@@UNKNOWN@@`` token (and hence padding tokens too
            because they are basically the same).
        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size, )`` containing cross entropy loss of captions, summed
            across time-steps.
        """

        # shape: (batch_size, )
        target_lengths = torch.sum(target_mask, dim=-1).float()

        # shape: (batch_size, )
        return target_lengths * sequence_cross_entropy_with_logits(
            logits, targets, target_mask, average=None
        )

        # inference
        # train the model and check dataloader what the input caption is: start token + a man + ... --> output: a man + ...

    def _decode_step(
        self,
        image_features: torch.Tensor,
        previous_predictions: torch.Tensor,
        states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Given image features, tokens predicted at previous time-step and LSTM states of the
        :class:`~updown.modules.updown_cell.UpDownCell`, take a decoding step. This is also
        called by the beam search class.
        Parameters
        ----------
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes, image_feature_size)``.
        previous_predictions: torch.Tensor
            A tensor of shape ``(batch_size * net_beam_size, )`` containing tokens predicted at
            previous time-step -- one for each beam, for each instances in a batch.
            ``net_beam_size`` is 1 during teacher forcing (training), ``beam_size`` for regular
            :class:`allennlp.nn.beam_search.BeamSearch` and ``beam_size * num_states`` for
            :class:`updown.modules.cbs.ConstrainedBeamSearch`
        states: [Dict[str, torch.Tensor], optional (default = None)
            LSTM states of the :class:`~updown.modules.updown_cell.UpDownCell`. These are
            initialized as zero tensors if not provided (at first time-step).
        """
        net_beam_size = 1

        # Expand and repeat image features while doing beam search (during inference).
        if not self.training and image_features.size(0) != previous_predictions.size(0):

            batch_size, num_boxes, image_feature_size = image_features.size()
            net_beam_size = int(previous_predictions.size(0) / batch_size)

            # Add (net) beam dimension and repeat image features.
            image_features = image_features.unsqueeze(1).repeat(1, net_beam_size, 1, 1)

            # shape: (batch_size * net_beam_size, num_boxes, image_feature_size)
            image_features = image_features.view(
                batch_size * net_beam_size, num_boxes, image_feature_size
            )

        # shape: (batch_size * net_beam_size, )
        current_input = previous_predictions

        # shape: (batch_size * net_beam_size, embedding_size)
        token_embeddings = self._embedding_layer(current_input)

        # shape: (batch_size * net_beam_size, hidden_size)
        updown_output, states = self._updown_cell(image_features, token_embeddings, states)

        # shape: (batch_size * net_beam_size, vocab_size)
        updown_output = self._output_projection(updown_output)
        output_logits = self._output_layer(updown_output)

        # Return logits while training, to further calculate cross entropy loss.
        # Return logprobs during inference, because beam search needs them.
        # Note:: This means NO BEAM SEARCH DURING TRAINING.
        outputs = output_logits if self.training else self._log_softmax(output_logits)

        return outputs, states  # type: ignore