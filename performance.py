#!/usr/bin/env python

from modules.models import load_model, GAIARegressor, SDSSRegressor
from torch.utils.data import DataLoader
from modules.utils import load_data
import matplotlib.pyplot as plt
import argparse

TRAIN_PATH = 'data/train'
VALID_PATH = 'data/valid'
MODELS = ['gaia', 'sdss']
ONE_TO_ONE = 'One-to-One Regression Line'
REGLINE = 'Regression Line'
COLOR = 'springgreen'

def sdss_to_gaia(model: GAIARegressor, loader: DataLoader) -> None:
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Prediction vs Ground Truth Plots')
    fig.supxlabel('Model Prediction')
    fig.supylabel('Ground Truth')
    
    axs[0].set_xlabel(r'$G$ predicted')
    axs[0].set_ylabel(r'$G$ ground truth')
    axs[0].set_title(r'$G$ Magnitude')

    axs[1].set_xlabel(r'$B_P$ predicted')
    axs[1].set_ylabel(r'$B_P$ ground truth')
    axs[1].set_title(r'$B_P$ Magnitude')
    
    axs[2].set_xlabel(r'$R_P$ predicted')
    axs[2].set_ylabel(r'$R_P$ ground truth')
    axs[2].set_title(r'$R_P$ Magnitude')
    
    m1, M1 = 0, 0
    m2, M2 = 0, 0
    m3, M3 = 0, 0

    for x, y in loader:
        pred = model(x)
        pred = pred.detach().numpy().T
        y = y.detach().numpy().T

        g_p, g_y = pred[0], y[0]
        bp_p, bp_y = pred[1], y[1]
        rp_p, rp_y = pred[2], y[2]

        m1 = min(m1, min(g_y)) if m1 != 0 else min(g_y)
        M1 = max(M1, max(g_y))
        m2 = min(m2, min(bp_y)) if m2 != 0 else min(bp_y)
        M2 = max(M2, max(bp_y))
        m3 = min(m3, min(rp_y)) if m3 != 0 else min(rp_y)
        M3 = max(M3, max(rp_y))

        axs[0].scatter(g_p, g_y, marker='.', color=COLOR)
        axs[1].scatter(bp_p, bp_y, marker='.', color=COLOR)
        axs[2].scatter(rp_p, rp_y, marker='.', color=COLOR)
    
    axs[0].plot([m1, M1], [m1, M1], c='r', linestyle='--', label=ONE_TO_ONE)
    axs[0].legend(loc='best')
    axs[1].plot([m2, M2], [m2, M2], c='r', linestyle='--', label=ONE_TO_ONE)
    axs[1].legend(loc='best')
    axs[2].plot([m3, M3], [m3, M3], c='r', linestyle='--', label=ONE_TO_ONE)
    axs[2].legend(loc='best')

    plt.show()

def gaia_to_sdss(model: SDSSRegressor, loader: DataLoader) -> None:
    fig, axs = plt.subplots(1, 5)
    fig.suptitle('Prediction vs Ground Truth Plots')
    fig.supxlabel('Model Prediction')
    fig.supylabel('Ground Truth')
    
    axs[0].set_xlabel(r'$u$ predicted')
    axs[0].set_ylabel(r'$u$ ground truth')
    axs[0].set_title(r'$u$ Magnitude')

    axs[1].set_xlabel(r'$g$ predicted')
    axs[1].set_ylabel(r'$g$ ground truth')
    axs[1].set_title(r'$g$ Magnitude')
    
    axs[2].set_xlabel(r'$r$ predicted')
    axs[2].set_ylabel(r'$r$ ground truth')
    axs[2].set_title(r'$r$ Magnitude')
    
    axs[3].set_xlabel(r'$i$ predicted')
    axs[3].set_ylabel(r'$i$ ground truth')
    axs[3].set_title(r'$i$ Magnitude')

    axs[4].set_xlabel(r'$z$ predicted')
    axs[4].set_ylabel(r'$z$ ground truth')
    axs[4].set_title(r'$z$ Magnitude')
    
    m1, M1 = 0, 0
    m2, M2 = 0, 0
    m3, M3 = 0, 0
    m4, M4 = 0, 0
    m5, M5 = 0, 0

    for x, y in loader:
        pred = model(x)
        pred = pred.detach().numpy().T
        y = y.detach().numpy().T

        u_p, u_y = pred[0], y[0]
        g_p, g_y = pred[1], y[1]
        r_p, r_y = pred[2], y[2]
        i_p, i_y = pred[3], y[3]
        z_p, z_y = pred[4], y[4]

        m1 = min(m1, min(u_y)) if m1 != 0 else min(u_y)
        M1 = max(M1, max(u_y))
        m2 = min(m2, min(g_y)) if m2 != 0 else min(g_y)
        M2 = max(M2, max(g_y))
        m3 = min(m3, min(r_y)) if m3 != 0 else min(r_y)
        M3 = max(M3, max(r_y))
        m4 = min(m4, min(i_y)) if m4 != 0 else min(i_y)
        M4 = max(M4, max(i_y))
        m5 = min(m5, min(z_y)) if m5 != 0 else min(z_y)
        M5 = max(M5, max(z_y))
        
        axs[0].scatter(u_p, u_y, marker='.', color=COLOR)
        axs[1].scatter(g_p, g_y, marker='.', color=COLOR)
        axs[2].scatter(r_p, r_y, marker='.', color=COLOR)
        axs[3].scatter(i_p, i_y, marker='.', color=COLOR)
        axs[4].scatter(z_p, z_y, marker='.', color=COLOR)
    
    axs[0].plot([m1, M1], [m1, M1], c='r', linestyle='--', label=REGLINE)
    axs[0].legend(loc='best')
    axs[1].plot([m2, M2], [m2, M2], c='r', linestyle='--', label=REGLINE)
    axs[1].legend(loc='best')
    axs[2].plot([m3, M3], [m3, M3], c='r', linestyle='--', label=REGLINE)
    axs[2].legend(loc='best')
    axs[3].plot([m4, M4], [m4, M4], c='r', linestyle='--', label=REGLINE)
    axs[3].legend(loc='best')
    axs[4].plot([m5, M5], [m5, M5], c='r', linestyle='--', label=REGLINE)
    axs[4].legend(loc='best')

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--model', default='gaia', type=str, help='what model to visualize performance for')
    parser.add_argument('-t', '--training', action='store_true', help='visuals for training data instead')

    args = parser.parse_args()

    assert args.model in MODELS, '%s is an unknown model type' % args.model

    model = load_model(args.model)
    model.eval()
    data_loader = load_data(TRAIN_PATH if args.training else VALID_PATH, model=args.model)

    if args.model == 'gaia':
        sdss_to_gaia(model, data_loader)
    elif args.model == 'sdss':
        gaia_to_sdss(model, data_loader)
    else:
        raise ValueError('%s is not a recognized model type' % args.model)
    
