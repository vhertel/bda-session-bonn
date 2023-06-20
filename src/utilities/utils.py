import csv
import json
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist


# ----------------------------------------------------------------------------------------------------------------------
#   CLASSES
# ----------------------------------------------------------------------------------------------------------------------

class Summary(Enum):
    """Enumeration for summary types."""
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """
    Computes and stores the average and current value.

    Args:
        name (str): Name of the meter.
        fmt (str): Format string for displaying values.
        summary_type (Summary): Type of summary to be displayed.

    Reference:
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        """Resets the meter values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter values.

        Args:
            val (float): Current value.
            n (int): Number of occurrences of the value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        """Applies all-reduce operation to synchronize values across distributed training."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        """Returns a string representation of the meter."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        """Returns a string representation of the meter summary."""
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('Invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """
    Displays and manages the progress of training or evaluation.

    Args:
        num_batches (int): Total number of batches.
        meters (list): List of AverageMeter objects.
        prefix (str): Prefix for displaying additional information.
    """

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        """
        Displays the progress for a given batch.

        Args:
            batch (int): Current batch number.
        """
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        """Displays the summary of all meters."""
        entries = ["   *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        """
        Returns the format string for displaying the batch number.

        Args:
            num_batches (int): Total number of batches.

        Returns:
            str: Format string for displaying the batch number.
        """
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# ----------------------------------------------------------------------------------------------------------------------
#   METHODS
# ----------------------------------------------------------------------------------------------------------------------

def load_json(json_filename):
    """Loads json file and stores content in Python object.

    Args:
        json_filename: path to json file

    Returns:
        file_content: content of json file as dictionary
    """
    with open(json_filename) as f:
        file_content = json.load(f)

    return file_content


def save_logs_to_csv(logs, filename):
    """
    Save the training logs to a CSV file.

    Args:
        logs        (list):         List of lists containing the training logs for each epoch. Each inner list
                                    contains the epoch number, loss, F1 macro, and F1 micro scores
        filename    (str or Path):  Name or Path object representing the CSV file to save the logs

    """
    header = ['Epoch', 'Loss', 'F1 Macro', 'F1 Micro']
    # Convert Path object to string if necessary
    if isinstance(filename, Path):
        filename = str(filename)
    # Open the file in write mode and create a CSV writer
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(header)
        # Write the logs for each epoch
        for log in logs:
            writer.writerow(log)


def plot(y_arr, pred_arr, path):
    # configure plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 20))
    # tile index in batch
    tile_index = 5
    # plot
    axes[0].imshow(y_arr[tile_index])
    axes[0].set_title('Target')
    axes[1].imshow(pred_arr[tile_index])
    axes[1].set_title('Output')
    # save figure
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_logs(df, path):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)

    # Subplot 1
    axes[0].plot(df.index + 1, df['Loss'])
    axes[0].set_ylabel('Loss')
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Subplot 2
    axes[1].plot(df.index + 1, df['F1 macro'])
    axes[1].set_ylabel('F1 macro')
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].set_ylim(0, 1)

    # Subplot 3
    axes[2].plot(df.index + 1, df['F1 micro'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 micro')
    axes[2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[2].set_ylim(0, 1)

    # save figure
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
