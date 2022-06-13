#!/usr/bin/env python

from modules.models import load_model
from colorama import init, Fore
from typing import Union
import pandas as pd
from os import path
import sys
import torch
import numpy as np

sys.tracebacklimit = 0

init(convert=True, autoreset=True)

def parse_data(data: str) ->  Union[torch.Tensor, tuple]:
    if ',' in data:
        assert data.count(',') == 4, Fore.RED + 'Data must have exactly 5 SDSS psf magnitudes'\
                                                ' separated by exactly 4 commas'
        mags = data.split(',')
        assert '' not in mags, Fore.RED + 'All 5 SDSS psf magnitudes need to be specified'

        float_mags = list()
        for mag in mags:
            try:
                float_mags.append(float(mag))
            except ValueError:
                print(Fore.RED + 'all magnitudes must be floats. %s is not a float' % mag)
                sys.exit(1)

        return torch.tensor([float_mags])

    assert path.exists(data), Fore.RED + 'specified relative path does not exist: %s' % data
    assert path.isfile(data), Fore.RED + 'specified relative path is not a file'
    assert data.endswith('.csv'), Fore.RED + 'source file must be a csv'

    df: pd.DataFrame = None
    try:
        df = pd.read_csv(data)
    except Exception:
        sys.tracebacklimit = 10
        print(Fore.RED + 'Error occured while reading from csv')
        sys.exit(1)
    
    # tests
    df2 = df.dropna()
    assert df.shape == df2.shape, Fore.RED + 'dataframe has blank fields'
    colnames = ['u', 'g', 'r', 'i', 'z']
    for col in colnames:
        assert col in df.keys(), Fore.RED + 'dataframe does not contain the required fields: %s' % col

    main_cols = df[colnames]
    t = torch.tensor(main_cols.to_numpy().astype(np.float32))
    return (df, t)


if __name__ == '__main__':
    print(Fore.CYAN + ' -------- Convert GAIA magnitudes to SDSS psf magnitudes -------- ')
    print(Fore.YELLOW + 'Enter data: (relative filepath / comma separated data) ->')
    data = input('')
    
    # parse data to determine operation
    output = parse_data(data)

    model = load_model('gaia')

    if type(output) == tuple:
        # dataframe (must append cols to df)
        df, cols = output
        prediction = model(cols)
        preds = prediction.detach().numpy().T
        g = preds[0]
        bp = preds[1]
        rp = preds[2]

        df['gaia_g'] = g
        df['gaia_bp'] = bp
        df['gaia_rp'] = rp

        df.to_csv('./gaia_added.csv', index=False)

    else:
        # single output
        prediction = model(output)
        arr = prediction.detach().numpy()[0]
        print(Fore.CYAN + 'g: %.3f' % arr[0])
        print(Fore.CYAN + 'bp: %.3f' % arr[1])
        print(Fore.CYAN + 'rp: %.3f' % arr[2])

