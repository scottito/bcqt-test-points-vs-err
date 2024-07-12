
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

from lmfit import Model, Parameters
from matplotlib import gridspec

def display_resonator_data(freq, magn, phase):
    mosaic = """  AB
                  CC
                  """

    fig, axes = plt.subplot_mosaic(mosaic,figsize=(8,8))
    ax1, ax2, ax3 = axes.values()
    axes = (ax1, ax2, ax3) # convert default dict to list

    real = np.real(magn * np.exp(1j * phase))
    imag = np.imag(magn * np.exp(1j * phase))

    ax1.plot(freq/1e9, magn, marker='x', color='k')
    ax2.plot(freq/1e9, phase, marker='x', color='r')
    ax3.scatter(real, imag, s=50, color='g')

    ax1.set_title("Magnitude")
    ax1.set_xlabel("Frequency [GHz]")
    ax1.set_ylabel("Amplitude [lin]")

    ax2.set_title("Phase")
    ax2.set_xlabel("Frequency [GHz]")
    ax2.set_ylabel("Phase [rad]")

    ax3.set_title("Complex")
    ax3.set_xlabel("Real")
    ax3.set_ylabel("Imag")

    fig.tight_layout()

    return fig, axes


def load_and_prep_csv(file_path):
    df = pd.read_csv(file_path, sep=",", names=['Frequency','dBm','Phase'])

    df['Ampl'] = 10**(df['dBm']/20)
    df['Phase'] = np.unwrap(np.deg2rad(df['Phase']))
    df['Complex'] = df['Ampl'] * np.exp(1j*df['Phase'] )

    freq = df['Frequency']
    magn = df['Ampl']
    phase = df['Phase']
    cmplx = df['Complex']

    return freq, magn, phase, cmplx


def quick_fit_to_magnitude(lmfit_model, freq, magn, **kwargs):
    if 'f0' not in kwargs.keys():
        f0_guess_idx = magn.argmin()  # reasonable starting point
        f0_guess = freq[f0_guess_idx]
    else:
        f0_guess = kwargs['f0']   

    params = Parameters()
    params.add('f0', f0_guess, min=f0_guess-25e3, max=f0_guess+25e3 )  # do not want this to vary by much
    params.add('Q',  value=1e4, min=1e3, max=1e9) # sensible lower/upper bounds
    params.add('Qc', value=1e4, min=1e3, max=1e9)
    params.add('A', min=1e-5, max=1e5)

    for key, val in kwargs.items():
        if key != 'f0':   
            print( "adding: {} = {}".format(key, val))
            params.add(key, val)

    result_magn = lmfit_model.fit(magn, params, f=freq)

    return result_magn


def col_by_n_plots(N, cols):
    """ creates a pre-structured figure of N plots in 'cols' columns """
    rows = int(math.ceil(N / cols))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure()
    axes = []

    for n in range(N):
        ax = fig.add_subplot(gs[n])
        axes = [*axes, ax]

    return fig, axes    


def s21_db_to_lin(np_array):
    """ I always forget this conversion... factor of 20 for s21 since it's a voltage measurement """
    return 10**(np_array/20)


