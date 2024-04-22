#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import os
import argparse
import smdebug.pytorch as smd
from time import time


def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    if hook:
        hook.set_mode(smd.modes.EVAL)
    model.eval()
    running_loss=0.0
    running_corrects=0
    running_samples=0
    for index,(inps, labels) in enumerate(test_loader):
            inps=inps.to(device)
            labels=labels.to(device)
            output=model(inps)
            loss=criterion(output, labels)
            
            
            _,preds=torch.max(output, dim=1)
            running_loss+= loss.item()*inps.size(0)
            running_corrects+=torch.sum(preds==labels.data).item()
            running_samples+=len(inps)
            
    epoch_loss=running_loss/running_samples
    epoch_acccuracy=running_corrects/running_samples
    print("\nTest set: Average loss: {:.4f}, Accuracy: ({:.2f}%) \n".format( epoch_loss, epoch_acccuracy))
    return model
    

def train(model, train_loader, criterion, optimizer, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    if hook:
        hook.set_mode(smd.modes.TRAIN)
    best_loss=1e6
    for e in range(args.epochs):
        print(f"Epoch : {e}")
        model.train()
        running_loss=0.0
        running_corrects=0
        running_samples=0
        for index,(inps, labels) in enumerate(train_loader):
            #print(index, inps.shape, labels.shape)
            inps=inps.to(device)
            labels=labels.to(device)
            output=model(inps)
            loss=criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _,preds=torch.max(output, dim=1)
            running_loss+= loss.item()*inps.size(0)
            running_corrects+=torch.sum(preds==labels.data).item()
            running_samples+=len(inps)
            
            if running_samples%(args.batch_size)==0:
                accuracy=running_corrects/running_samples
                print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(train_loader.dataset),
                            100.0 * (running_samples / len(train_loader.dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy
                        )
                     )
        
          
        epoch_loss= running_loss/ running_samples
        epoch_accuracy=running_corrects/running_samples
        if epoch_loss<best_loss:
            print("best loss :", epoch_loss)
            print(f"storing latest model {e}")
            #torch.save(model.state_dict(), "food_classifier_best.pt")
            best_loss=epoch_loss
    
        print("Epoch : {} Finished  Loss: {:.2f} Accuracy: {:.2f}%".format(e, epoch_loss, epoch_accuracy))
        print("++++"*25)
    return model
    
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

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_data_path=os.path.join(data,'train')
    test_data_path=os.path.join(data,'test')
    train_data_transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    train_dataset=datasets.ImageFolder(root=train_data_path,transform=train_data_transform)
    print(f"train_dataset classes {train_dataset.classes}")
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_data_transform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    test_dataset=datasets.ImageFolder(root=test_data_path,transform=test_data_transform)
    print(f"test_dataset classes {test_dataset.classes}")
        
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    
    
    return train_loader, test_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    print(f" data  {args.data} lr {args.lr}, epoch : {args.epochs} batch size {args.batch_size} , model path {args.model_dir}, output dir {args.output_dir}" )
    model=net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    model = model.to(device)
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader, test_loader = create_data_loaders(args.data, args.batch_size)
    
    print(f"start training")
    st=time()
    model=train(model, train_loader, loss_criterion, optimizer, device, hook)
    end=time()
    print(f"Training Time {end-st} sec")
    
    
    '''
    TODO: Test the model to see its accuracy
    '''
    print("Starting Testing")
    st=time()
    test(model, test_loader, loss_criterion, device, hook)
    end=time()
    print(f"Testing Time {end-st} sec")
    
    '''
    TODO: Save the trained model
    '''
    print("saving model !!")
    torch.save(model, os.path.join(args.model_dir,'model.pth'))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    
    
    parser.add_argument('--early-stopping-rounds',
                        type=int,
                        default=10)
    parser.add_argument('--data', type=str,
                        default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir',
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir',
                        type=str,
                        default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    
    
    args=parser.parse_args()
    
    main(args)
