import json
html_string = """
<font size = 40px face = 'Courier New'>"""         

with open('nocaps_val_image_info.json') as json_file:
    data = json.load(json_file)
with open('updown_predictions_nocaps_val.json') as caption_file:
    captions_file = json.load(caption_file)
    table = "<table width='100%' border='0' cellpadding='10' cellspacing='0'>"
    tr = ""
    for i in range(0, len(data['images']), 5):
        td = ""
        tr += "<tr>"
        for j in range(i, i + 5):
            td = ""
            image_url = data['images'][j]['coco_url']
            caption = captions_file[j]['caption']
            td += "<td align='center' valign='center' >"
            td += "<img align='center' width='100%' height='auto' src={} />".format(image_url)
            td += "<br /><br />"
            td += "<b>{}</b>".format(caption)
            td += "</td>"
            tr += td
        tr += "</tr>"
    table += tr
    table += "</table>"
    html_string += table
        



with open('no_caps_val.html', 'w') as f:
    f.write(html_string)


