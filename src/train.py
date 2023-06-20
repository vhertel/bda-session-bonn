import time
from pathlib import Path

import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.utilities.DatasetFromStruct import DatasetFromStruct
from src.utilities.utils import AverageMeter, ProgressMeter, Summary, save_logs_to_csv, plot_logs

# print setup
print(f'\nUsing PyTorch version {torch.__version__}.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}.')

CONFIG = {'epochs': 10,
          'learning_rate': 0.0005,
          'batch_size': 32,
          'train_dir': 'D:/Bonn/train',
          'val_dir': 'D:/Bonn/val',
          'model': 'pre_trained.pth.tar'}


def epoch_eval(mode, model, data_loader, loss_func, epoch, optimizer=None):
    """This function handles all processing steps for the evaluation of one epoch. Depending on the evaluation mode
    (training or validation), the gradient propagation is set accordingly. The function returns the average F1 scores
    (macro and micro) as well as the average loss for the current epoch.

    Args:
        mode          (String):                         Current training set (train or val)
        data_loader   (torch.utils.data.DataLoader):    Iterable to pass samples in minibatches, reshuffle data, and use
                                                        Python’s multiprocessing to speed up data retrieval
        model         (torch.nn.Module):                Model architecture to be trained
        loss_func     (torch.nn loss function):         Loss function to be applied for classification tasks
        epoch         (int):                            Current epoch number
        optimizer     (torch.optim optimizer):          Optimization algorithm to be used for training

    Returns:
         f1_macro.avg (float):                          average macro F1 score of epoch
         f1_micro.avg (float):                          average micro F1 score of epoch
         losses.avg   (float):                          average loss of epoch
    """

    if mode == 'train':
        # enable gradient propagation for training
        torch.set_grad_enabled(True)
        # format monitoring of training variables
        prefix = f'   Epoch: [{epoch}]'
    elif mode == 'val':
        # disable gradient propagation for validation
        torch.set_grad_enabled(False)
        # format monitoring of training variables
        prefix = '   Val: '
    else:
        raise Exception('Invalid set of input data.')

    # initialize monitoring of training variables
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', ':6.3f', Summary.NONE)
    loss = AverageMeter('Loss', ':.4e')
    f1_macro = AverageMeter('F1_macro', ':6.2f')
    f1_micro = AverageMeter('F1_micro', ':6.2f')
    progress = ProgressMeter(len(data_loader), [batch_time, data_time, loss, f1_macro, f1_micro], prefix=prefix)

    # Softmax function rescaling tensor values to lie in the range [0, 1] and sum to 1
    softmax = torch.nn.Softmax(dim=1)

    # set start time for batch
    start = time.time()
    # loop through batches
    for batch, (X, y) in enumerate(data_loader):
        # load data to GPU as PyTorch tensors
        X, y = X.to(device=device), y.to(device=device)
        # measure data loading time
        data_time.update(time.time() - start)

        # model inference
        pred = model(X)

        # compute batch loss for classification task
        batch_loss = loss_func(pred, y)

        if mode == 'train':
            # reset gradients from previous batch
            optimizer.zero_grad()
            # compute backward gradients of loss function
            batch_loss.backward()
            # adjust learning weights based on gradients
            optimizer.step()

        # compute segmentation output of pre disaster image
        pred_cls = torch.argmax(softmax(pred), dim=1)

        # get actual size of batch (last batch of epoch might not match the specified batch size in train config)
        batch_size = X.size(0)
        # update epoch loss with current batch loss
        loss.update(batch_loss.item(), batch_size)

        # compute performance metrics of batch
        y_flat = y.detach().cpu().numpy().flatten()
        pred_cls_flat = pred_cls.detach().cpu().numpy().flatten()
        f1_macro.update(f1_score(y_flat, pred_cls_flat, average='macro'), batch_size)
        f1_micro.update(f1_score(y_flat, pred_cls_flat, average='micro'), batch_size)

        # measure batch time
        batch_time.update(time.time() - start)
        # reset start time next batch
        start = time.time()

        # print status update every batch
        if batch % 1 == 0:
            progress.display(batch + 1)

    # display monitoring of training variables
    progress.display_summary()

    # return average metrics and loss for epoch
    return f1_macro.avg, f1_micro.avg, loss.avg


def main():
    """Main function to start process."""

    # ------------------------------------------------------------------------------------------------------------------
    #   MODEL
    # ------------------------------------------------------------------------------------------------------------------
    # load model checkpoint
    checkpoint_dir = Path.cwd().parent.joinpath('res', 'models', CONFIG['model'])
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model = smp.Unet(in_channels=checkpoint['in_channels'], classes=checkpoint['classes']).to(device=device)
    # load model to continue with inference
    model.load_state_dict(checkpoint['state_dict'])
    print(f'\nLoaded model from {str(checkpoint_dir)}')

    # ------------------------------------------------------------------------------------------------------------------
    #   DATA
    # ------------------------------------------------------------------------------------------------------------------

    # get Path object to training and validation data
    train_dir = Path(CONFIG['train_dir'])
    val_dir = Path(CONFIG['val_dir'])
    # load average mean and standard deviation of dataset
    data_statistics = checkpoint['statistics']

    # create DatasetFromStruct instance (torch.utils.data.Dataset) for training and validation data
    train_set = DatasetFromStruct(train_dir, data_statistics, normalize=True)
    val_set = DatasetFromStruct(val_dir, data_statistics, normalize=True)
    # combine datasets and a sampler to provide an iterable over the given dataset
    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    # print status update
    print(f'Train data length: {len(train_set)}')
    print(f'Validation data length: {len(val_set)}\n')

    # ------------------------------------------------------------------------------------------------------------------
    #   TRAINING
    # ------------------------------------------------------------------------------------------------------------------

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    # loss functions for semantic segmentation
    loss_func = nn.CrossEntropyLoss().to(device=device)

    # empty list for validation logs
    logs = []

    for epoch in range(1, CONFIG['epochs'] + 1):
        # turn on gradient tracking for learning
        model.train()
        _, _, _ = epoch_eval('train', model, train_loader, loss_func, epoch, optimizer)
        # set model to evaluation mode
        model.eval()
        f1_macro, f1_micro, loss = epoch_eval('val', model, val_loader, loss_func, epoch)
        # append current epoch's logs to the logs list
        logs.append([epoch, loss, f1_macro, f1_micro])
        # save model
        torch.save({'state_dict': model.state_dict(),
                    'in_channels': checkpoint['in_channels'],
                    'classes': checkpoint['classes'],
                    'epochs': CONFIG['epochs'],
                    'statistics': data_statistics,
                    'f1_macro': f1_macro,
                    'f1_micro': f1_micro,
                    'loss': loss,
                    }, Path.cwd().parent.joinpath('res', 'models',
                                                  f'{epoch}-{round(f1_macro, 4)}-{round(loss, 4)}.pth.tar'))
    # save log
    save_logs_to_csv(logs, Path.cwd().parent.joinpath('res', 'logs', 'logs.csv'))
    plot_logs(pd.DataFrame(logs, columns=['Epoch', 'Loss', 'F1 macro', 'F1 micro']),
              Path.cwd().parent.joinpath('res', 'logs', 'logs.png'))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
