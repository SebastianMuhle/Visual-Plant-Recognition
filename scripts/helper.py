#
# BSD 3-Clause License
#
#
# Copyright (c) 2022, Sebasitan Muhle, Dominik Muhle.
# All rights reserved.

import scipy.io
import pandas as pd
from pathlib import Path

# https: // gist.github.com/techedlaksh/9001039bf54ba9d8aec3ad7f5d8bfd08


def convert_mat_to_csv(filename):
    path = Path(filename)
    mat = scipy.io.loadmat(path)
    mat = {k: v for k, v in mat.items() if k[0] != '_'}
    data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
    data.to_csv(path.with_suffix('.csv'))
