import time
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.utilities.DatasetFromStruct import DatasetFromStruct
from src.utilities.utils import AverageMeter, ProgressMeter, Summary, plot

# print setup
print(f'\nUsing PyTorch version {torch.__version__}.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}.')

CONFIG = {'batch_size': 16,
          'test_dir': 'D:/Datasets/xBD_Bonn/test',
          'model': 'res/models/pre_trained.pth.tar'}


def inference(loader, model):
    """This function applies a trained model to test data. Depending on the input configurations, the dataset is either
    read from a data split JSON file or from the directory structure. Metrics and output visualizations are created
    depending on the config.

    Args:
        loader        (torch.utils.data.DataLoader):    Iterable to pass samples in minibatches, reshuffle data, and use
                                                        Python’s multiprocessing to speed up data retrieval
        model         (torch.nn.Module):                Model architecture to be trained
    """

    # Softmax function rescaling tensor values to lie in the range [0, 1] and sum to 1
    softmax = torch.nn.Softmax(dim=1)
    # set model to evaluation mode
    model.eval()

    # initialize monitoring of training variables
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', ':6.3f', Summary.NONE)
    f1_macro = AverageMeter('F1_macro', ':6.2f')
    f1_micro = AverageMeter('F1_micro', ':6.2f')
    progress = ProgressMeter(len(loader), [batch_time, data_time, f1_macro, f1_micro], prefix='   Inference: ')

    # disable gradient calculation to reduce memory consumption
    with torch.no_grad():
        # set start time for batch
        start = time.time()

        # loop through batches
        for batch, (X, y, paths) in enumerate(loader):
            # load data to GPU as PyTorch tensors
            X, y = X.to(device=device), y.to(device=device)
            # measure data loading time
            data_time.update(time.time() - start)

            # model inference
            pred = model(X)

            # compute segmentation output of pre disaster image
            pred_cls = torch.argmax(softmax(pred), dim=1)

            # get actual size of batch (last batch of epoch might not match the specified batch size in train config)
            batch_size = X.size(0)
            # compute performance metrics of batch
            y_arr = y.detach().cpu().numpy()
            pred_arr = pred_cls.detach().cpu().numpy()
            f1_macro.update(f1_score(y_arr.flatten(), pred_arr.flatten(), average='macro'), batch_size)
            f1_micro.update(f1_score(y_arr.flatten(), pred_arr.flatten(), average='micro'), batch_size)

            # measure batch time
            batch_time.update(time.time() - start)
            # reset start time next batch
            start = time.time()

            # plot tiles
            Path.cwd().parent.joinpath('res', 'output').mkdir(exist_ok=True)
            plot(y_arr, pred_arr, paths)

            # print status update every batch
            if batch % 1 == 0:
                progress.display(batch + 1)

    # display monitoring of training variables
    progress.display_summary()


def main():
    """Main function to start process."""

    # ------------------------------------------------------------------------------------------------------------------
    #   MODEL
    # ------------------------------------------------------------------------------------------------------------------
    # load model checkpoint
    checkpoint_dir = Path.cwd().parent.joinpath(CONFIG['model'])
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model = smp.Unet(in_channels=checkpoint['in_channels'], classes=checkpoint['classes']).to(device=device)
    # load model to continue with inference
    model.load_state_dict(checkpoint['state_dict'])
    print(f'\nLoaded model from {str(checkpoint_dir)}')

    # ------------------------------------------------------------------------------------------------------------------
    #   DATA
    # ------------------------------------------------------------------------------------------------------------------
    # get Path object to test data
    test_dir = Path(CONFIG['test_dir'])
    # load average mean and standard deviation of dataset
    data_statistics = checkpoint['statistics']

    # create DatasetFromStruct instance (torch.utils.data.Dataset) for training and validation data
    # normalize features to be on a similar scale to improve performance and training stability
    test_set = DatasetFromStruct(test_dir, data_statistics, normalize=True)
    # combine datasets and a sampler to provide an iterable over the given dataset
    test_loader = DataLoader(test_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    # print status update
    print(f'Test data length: {len(test_set)}')

    # ------------------------------------------------------------------------------------------------------------------
    #   INFERENCE
    # ------------------------------------------------------------------------------------------------------------------
    inference(test_loader, model)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
