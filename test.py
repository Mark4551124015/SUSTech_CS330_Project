from PIL import Image
import requests
import base64

img1 = './data/AnteriorSegment/couple/2/20240320fzx_20240320_195620_L_CASIA2_001_000.jpg'
img2 = './data/AnteriorSegment/single/20240320myc_20240320_192040_L_CASIA2_001_000.jpg'



img1 = Image.open(img1).convert('L')
img2 = Image.open(img2).convert('L')

img1_str = base64.b64encode(img1.tobytes()).decode()
img2_str = base64.b64encode(img2.tobytes()).decode()

res=requests.post('http://127.0.0.1:8888/couple', json={
    'img_1':img1_str,
    'shape_1':img1.size,
    'img_2':img2_str,
    'shape_2':img2.size})

print(res.content)