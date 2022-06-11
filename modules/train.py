#!/usr/bin/env python

from .models import model_factory, save_model, load_model
from .utils import load_data, accuracy
import torch
import argparse
import torch.utils.tensorboard as tb

TRAIN_PATH = 'data/train'
VALID_PATH = 'data/valid'

def train(args: argparse.Namespace) -> None:
    from os import path

    model = load_model(args.model) if args.retrain else model_factory[args.model]()
    learning_rate = args.learning
    epochs = args.epochs

    train_logger, valid_logger, global_step = None, None, 0
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
        global_step = 0

    # load data and initial setup
    train_data = load_data(TRAIN_PATH, args.model)
    valid_data = load_data(VALID_PATH, args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # trainers
    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(epochs):
        print('Epoch: %d' % epoch)

        model.train()

        for x, y in train_data:
            x = x.to(device)
            y = y.to(device)

            # forward pass through the network
            output = model(x)
            loss = loss_func(output, y)

            # update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update loss in tensorboard
            if args.log_dir is not None:
                train_logger.add_scalar('loss', loss, global_step=global_step)
                train_logger.add_scalar('accuracy', torch.mean(accuracy(output, y)).item(), global_step=global_step)
                global_step += 1
        
        if args.validate:
            model.eval()

            for x, y in valid_data:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)

                if args.log_dir is not None:
                    valid_logger.add_scalar('accuracy', torch.mean(accuracy(pred, y)).item(), global_step=global_step)

    save_model(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')

    parser.add_argument('-l', '--learning', default=0.001, type=float, help='learning rate for training')
    parser.add_argument('-e', '--epochs', default=10, type=int, help='number of epochs to train for')
    parser.add_argument('-m', '--model', default='gaia', type=str, help='what model type to train')
    parser.add_argument('-r', '--retrain', action='store_true', help='retrain existing model')
    parser.add_argument('-v', '--validate', action='store_true', help='simultaneously validate')

    args = parser.parse_args()
    train(args)