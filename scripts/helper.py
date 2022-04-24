#
# BSD 3-Clause License
#
#
# Copyright (c) 2022, Sebastian Muhle, Dominik Muhle.
# All rights reserved.

import scipy.io
import pandas as pd
from pathlib import Path
from progress import Progress as P


# https: // gist.github.com/techedlaksh/9001039bf54ba9d8aec3ad7f5d8bfd08
def convert_mat_to_csv(filename: Path):
    if not filename.exists():
        print("Could not find file. Please provide the correct path to the file.")
        return
    mat = scipy.io.loadmat(filename)
    mat = {k: v for k, v in mat.items() if k[0] != '_'}
    data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
    data.to_csv(filename.with_suffix('.csv'))


# https://medium.com/analytics-vidhya/progress-bar-python-for-jupyter-notebook-f68224955810
def progress_bar(num_batches):
    p = P(num_batches, mode='bar')
    split = P.Element("Split", 0)
    epoch = P.Element("Epoch", 0)
    batch = P.Element("Batch", 0, display_name='hide',
                      max_value=num_batches, value_display_mode=1)
    progress_time = P.ProgressTime(postfix="/epoch")
    loss = P.Element("Loss", 0)
    acc = P.Element("Acc", 0)

    # progress bar [====>    ]
    bar = P.Bar()

    # Formatting progress bar
    p = p(split)(epoch)(bar)(batch)(progress_time)(
        "- ")(loss)("- ")(acc)  # format progress bar

    # get final progress bar format
    p.get_format()

    return {'progress': p, 'bar': bar, 'split': split, 'epoch': epoch, 'batch': batch, 'time': progress_time, 'loss': loss, 'acc': acc}
