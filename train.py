import torch
import torch.utils.tensorboard as tb
import argparse
import os
import numpy as np

from utils import load_data
from agent_imitation.model import Action, save_model
from agent_imitation.vision_model import Vision, load_vision_model

def getRMSE(list_preds, list_targets, idx):
    predicted = np.array([x[idx] for x in list_preds])
    targets = np.array([x[idx] for x in list_targets])
    return np.sqrt(((predicted - targets)**2).mean())

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:',device)
    model = Action(normalize=True, inference=False).to(device) #Left infeerence to falsee because we are resizing in the vision model
    model.train(True)
    vision_model = load_vision_model()
    vision_model.to(device)
    vision_model.train(False)
    vision_model.eval()

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_data = load_data(args.train_path, batch_size=args.batch_size)
    # valid_data = load_data(args.valid_path, batch_size=args.batch_size)

    if args.logdir is not None:
        train_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'train_{}'.format(args.log_suffix)), flush_secs=1)
        # valid_logger = tb.SummaryWriter(log_dir=os.path.join(args.logdir, 'valid_{}'.format(args.log_suffix)), flush_secs=1)

    global_step = 0
    for e in range(args.epochs):
        print('Epoch:',e)
        all_targets = []
        all_predictions = []
        for batch in train_data:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            all_targets.append(batch[1].cpu().numpy())
            # print(images.shape)
            heatmaps, reshaped_images = vision_model(images)
            # print(heatmaps)
            combined_images = torch.cat((reshaped_images, torch.sigmoid(heatmaps)), 1)
            # print('shapes')
            # print(combined_images.shape)
            # print(combined_images.shape, images.shape, heatmaps.shape)
            pred = model(combined_images)
            all_predictions.append(pred.cpu().detach().numpy())
            l = loss(pred, labels.squeeze())
            # print(pred[0], labels.squeeze()[0])

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_logger.add_scalar('loss', l.cpu(), global_step=global_step)
            global_step += 1
        train_logger.add_scalar('RMSE_steer', getRMSE(all_predictions, all_targets, 0),global_step=e)
        train_logger.add_scalar('RMSE_acceleration', getRMSE(all_predictions, all_targets, 1),global_step=e)
        train_logger.add_scalar('RMSE_brake', getRMSE(all_predictions, all_targets, 2),global_step=e)
        save_model(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--valid_path', type=str)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_suffix', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()
    train(args)