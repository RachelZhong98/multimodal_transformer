import torch.nn as nn
from torchtext.vocab import GloVe
from 

class MultimodelTransformer(nn.Module):
    def __init__(self, vocabulary: Vocabulary, image_feature_size: int,num_attention_block: int,
                    caption_length: int, vocab_size: int, embedding_size: int, hidden_size: int,
                    feed_forward_dim: int, encode_dim: int, num_heads: int):
        self.image_feature_size = image_feature_size
        self.num_attention_block = num_attention_block
        self.caption_length = caption_length
        self.vocab_size = vocab_size
        self._vocabulary = vocabulary
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.feed_forward_dim = feed_forward_dim
        self.encode_dim = encode_dim
        self.num_heads = num_heads

        self.adapt_image_features = nn.Linear(self.image_feature_size, self.encode_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.encode_dim, nhead=8, dim_feedforward=self.feed_forward_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attention_block)
        mask = nn.Transformer._generate_square_subsequent_mask(self.caption_length)
        self.caption_lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=hidden_size, num_layers=1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.decode_dim, nhead=8, dim_feedforward=self.feed_forward_dim, tgt_mask = mask)
        self.decoder = nn.TransformerDecoder(decoder_layer)

        #check input dim 
        self.final = nn.Linear(self.caption_length, self.vocab_size)

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
            num_heads = _C.MODEL.NUM_HEADS
        )



    def encode(self, processed_image_features):
        return self.encoder(processed_image_features)
 
    def decode(self, embedded_words):
        return self.decoder(embedded_words)

    def embed(self, tokenized_words):
        return 

    def forward(self, image_features):
        processed_features = self.adapt_image_features(image_features)
        encoded_image_features = self.encode(processed_features)
        embedded_captions = self.embed()
        output = self.decoder(embedded_captions, encoded_image_features)
        output = nn.functional.softmax(self.final(output))
        return output
        

    def _initialize_glove(self) -> torch.Tensor:
        glove = GloVe(name="42B", dim=300)###
        glove_vectors = torch.zeros(self._vocabulary.get_vocab_size(), 300)

        for word, i in self._vocabulary.get_token_to_index_vocabulary().items():
            if word in glove.stoi:
                glove_vectors[i] = glove.vectors[glove.stoi[word]]

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