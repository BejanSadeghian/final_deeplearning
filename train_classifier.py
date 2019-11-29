import torch
import torch.utils.tensorboard as tb
import argparse
import os
import numpy as np

from PIL import Image
from torchvision import transforms

from utils import load_classifier_data, ClassifierData
from agent_manual.classifier_model import Classifier, save_model

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:',device)
    model = Classifier(normalize=True, inference=True).to(device)
    
    # loss = torch.nn.MSELoss()
    loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_data = load_classifier_data(args.train_path, batch_size=args.batch_size)
    valid_data = load_classifier_data(args.valid_path, batch_size=args.batch_size)

    if args.logdir is not None:
        train_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'train_{}'.format(args.log_suffix)), flush_secs=1)
        valid_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'valid_{}'.format(args.log_suffix)), flush_secs=1)

    global_step = 0
    for e in range(args.epochs):
        model.train(True)
        print('Epoch:',e)
        # all_targets = []
        # all_predictions = []

        for batch in train_data:
            images = batch[0].to(device)
            labels = batch[2].to(device) #Image at location 2
            
            # all_targets.append(batch[2].cpu().numpy())
            
            pred = model(images)
            # print(labels[1], pred[1])
            # print('batch', labels.shape, images.shape, pred.shape)

            # all_predictions.append(pred.cpu().detach().numpy())
            # print(labels.shape, pred.shape)
            l = loss(pred.cpu(), labels.cpu().float())
            
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_logger.add_scalar('loss', l.cpu(), global_step=global_step)
            global_step += 1
        # train_logger.add_scalar('RMSE_steer', getRMSE(all_predictions, all_targets, 0),global_step=e)
        # train_logger.add_scalar('RMSE_acceleration', getRMSE(all_predictions, all_targets, 1),global_step=e)
        # train_logger.add_scalar('RMSE_brake', getRMSE(all_predictions, all_targets, 2),global_step=e)
        model.eval()
        model.train(False)
        predictions = []
        actual = []
        for batch in valid_data:
            images = batch[0].to(device)
            labels = batch[2].to(device)

            pred = model(images)

            predictions.append(pred)
            actual.append(labels)
        
        predictions = torch.cat(predictions)
        actual = torch.cat(actual)
        
        valid_logger.add_scalar('Accuracy', (actual == predictions).cpu().detach().numpy().mean(), global_step=e)
        save_model(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--valid_path', type=str)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_suffix', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()
    train(args)