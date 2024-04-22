


#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import sys
from torchvision import datasets

import json
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'


try:
    logger.info("checking smedebug")
    import smdebug
    import smdebug.pytorch as smd
    logger.info("checking smedebug!!")
except:
    pass
    logger.info("failer")
    
try:
    logger.info("checking smedebug2")
    #import smdebug
    import smdebug.pytorch as smd
except:
    pass
    logger.info("failer2")


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    resnet18 = models.resnet18(pretrained=True)
    for param in resnet18.parameters():
        param.requires_grad=False
    
    num_features=resnet18.fc.in_features
    resnet18.fc = nn.Sequential(
                                nn.Linear(num_features,out_features=256),
                                nn.ReLU(),
                                nn.Linear(256,128),
                                nn.ReLU(),
                                nn.Linear(128,10)
                                )

    return resnet18



def model_fn(model_dir):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    model.eval()
    return model

def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    #if content_type == JPEG_CONTENT_TYPE: return io.BytesIO(request_body)
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    if content_type == JPEG_CONTENT_TYPE: return Image.open(io.BytesIO(request_body))
    logger.debug('SO loded JPEG content')
    # process a URL submitted to the endpoint
    
    if content_type == JSON_CONTENT_TYPE:
        #img_request = requests.get(url)
        logger.debug(f'Request body is: {request_body}')
        request = json.loads(request_body)
        logger.debug(f'Loaded JSON object: {request}')
        url = request['url']
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))
    
def predict_fn(input_object, model):
    logger.info('In predict fn')
    test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    logger.info("transforming input")
    input_object=test_transform(input_object)
    
    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))
        class_names=['chicken_curry','chicken_wings', 'fried_rice', 'grilled_salmon', 'hamburger', 'ice_cream', 'pizza', 'ramen', 'red_velvet_cake', 'steak', 'sushi','tacos']
        logger.info(f"prediction : {class_names[prediction]}")
    return prediction
    



