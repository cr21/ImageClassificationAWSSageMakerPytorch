


# #TODO: Import your dependencies.
# #For instance, below are some dependencies you might need if you are using Pytorch
# import subprocess

# #subprocess.call(['pip','install','nvgpu'])

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.models as models
# import torchvision.transforms as transforms
# import os
# import sys
# from torchvision import datasets

# import json
# import logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler(sys.stdout))
# JSON_CONTENT_TYPE = 'application/json'
# JPEG_CONTENT_TYPE = 'image/jpeg'


# try:
#     import smdebug
#     import smdebug.pytorch as smd
# except:
#     subprocess.call(['pip', 'install', 'smdebug'])
#     import smdebug
#     import smdebug.pytorch as smd
#     logger.info("Installing done")


# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger.info(f"device {device}")


# def net():
#     '''
#     TODO: Complete this function that initializes your model
#           Remember to use a pretrained model
#     '''
#     resnet18 = models.resnet18(pretrained=True)
#     for param in resnet18.parameters():
#         param.requires_grad=False
    
#     num_features=resnet18.fc.in_features
#     resnet18.fc = nn.Sequential(
#                                 nn.Linear(num_features,out_features=256),
#                                 nn.ReLU(),
#                                 nn.Linear(256,128),
#                                 nn.ReLU(),
#                                 nn.Linear(128,10)
#                                 )

#     return resnet18



# def model_fn(model_dir):
#     logger.info("in model_fn")
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     model = net()

#     with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
#         model.load_state_dict(torch.load(f))
#     model.to(device)
#     model.eval()
#     return model

# def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
#     logger.info('input_fn Enter')
#     if content_type == JPEG_CONTENT_TYPE: return Image.open(io.BytesIO(request_body))
#     # process a URL submitted to the endpoint
    
#     if content_type == JSON_CONTENT_TYPE:
#         #img_request = requests.get(url)
#         request = json.loads(request_body)
#         url = request['url']
#         img_content = requests.get(url).content
#         return Image.open(io.BytesIO(img_content))
    
#     raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))
    
# def predict_fn(input_object, model):
#     logger.info('In predict fn')
#     test_transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     input_object=test_transform(input_object)
    
#     with torch.no_grad():
#         prediction = model(input_object.unsqueeze(0))
#         class_names=['chicken_curry','chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'red_velvet_cake', 'steak', 'sushi','tacos']
#         #logger.info(f"prediction : {class_names[prediction]}")
#     return prediction
    




import subprocess
subprocess.call(['pip', 'install', 'smdebug'])

import smdebug
import json
import logging
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import requests


def model_fn(model_dir):
    
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
    
    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                                nn.Linear(num_features,out_features=256),
                                nn.ReLU(),
                                nn.Linear(256,128),
                                nn.ReLU(),
                                nn.Linear(128,10)
                                )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.to(device)
    model.eval()
    return model

def input_fn(request_body, content_type):
    if content_type == 'image/jpeg':
        return Image.open(io.BytesIO(request_body))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def predict_fn(input_object, model):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    input_object = transform(input_object)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction
