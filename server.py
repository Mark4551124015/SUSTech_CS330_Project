import flask
from flask import Flask, render_template, request, redirect, url_for
from flask import send_from_directory
import base64
import datetime
from PIL import Image
import io
from models.mynet import MyNet
from torchvision import transforms
import torch
from glob import glob
import os

male_path = '.saves/male/'
female_path = '.saves/female/'
unknown_path = '.saves/unknown/'

os.makedirs(male_path)
os.makedirs(female_path)
os.makedirs(unknown_path)


transform = transforms.Compose([
            transforms.Resize((400,400)),
            transforms.ToTensor()
        ])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
server=flask.Flask(__name__)
model = MyNet(n_classes=2).to(device)
model.eval()
checkpoint = torch.load('./ckpts/I_GOT_U_IN_MY_SITE/CKPT_Best.pth')
model.load_state_dict(checkpoint['model'])


def other_gender(gender):
    if gender == 'male':
        return 'female'
    if gender == 'female':
        return 'male'
    return 'unknown'

@server.route('/couple',methods=['post'])
def pair_couple():
    data = request.get_json()

    # 获取base64字符串
    base64_str = data.get('img_1')
    shape_1 = data.get('shape_1')
    image_data = base64.b64decode(base64_str)
    img1 = Image.frombuffer('L',shape_1,image_data,'raw','L',0,1)
    
    base64_str = data.get('img_2')
    shape_2 = data.get('shape_2')
    image_data = base64.b64decode(base64_str)
    img2 = Image.frombuffer('L',shape_2,image_data,'raw','L',0,1)
    
    img1=transform(img1).to(device)
    img2=transform(img2).to(device)
    
    img = torch.cat([img1,img2,img1],dim=0).unsqueeze(0)
    pred = model(img)
    pred = pred.cpu().detach().numpy()
    false_rate, true_rate = pred.flatten()
    matching = "%.4f"%(true_rate)
    return flask.jsonify({'status': 200, 'result': matching})

    
@server.route('/single',methods=['post'])
def single_search():
    data = request.get_json()
    base64_str = data.get('img')
    shape = data.get('shape')
    image_data = base64.b64decode(base64_str)
    img = Image.frombuffer('L',shape,image_data,'raw','L',0,1)
    matcher = transform(img).to(device)
    
    gender = data.get('gender')
    name = data.get('name')
    other_gen = other_gender(gender)
    if (gender!='male' and gender!='female'):
        return flask.jsonify({'status': 201 , 'result': 'Bad Gender'})
    # if (gender == 'male'):
    img.save(f'./saves/{gender}/{name}.png')
    matchings = glob(f'./saves/{other_gen}/*')
    X = []
    for path in matchings:
        other = Image.open(path).convert('L')
        other = transform(other).to(device)
        X.append(torch.cat([matcher,other,matcher],dim=0).unsqueeze(0))
    batch = torch.cat(X,dim=0)
    preds = model(batch)
    preds = preds.cpu().detach().numpy()
    matching = []
    if (len(preds) <= 0):
        return flask.jsonify({'status': 202, 'result': 'No Matchings'})
    for i in range(len(preds)):
        matching.append((matchings[i].split('/')[-1].split('.')[0], preds[i][1]))
    matching.sort(key=lambda x:x[1],reverse=True)
    best_name, best_score = matching[0]
    best_image = Image.open(f'./saves/{other_gen}/{best_name}.png')
    best_image_str = base64.b64encode(best_image.tobytes()).decode()
    best_score = "%.4f"%(best_score)
    return flask.jsonify({'status': 200, 'result': str(best_score), 'name': best_image, 'img': best_image_str})
    
server.run(port=8888, debug=True)