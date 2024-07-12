# %%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

from lmfit import Model, Parameters
from matplotlib import gridspec


def import_AWR_csv_as_df(csv_path, s_param="21", magn=True, unwrap=True, plot=True, verbose=False):
    # TODO: accept list of s params instead of single string
    s = s_param  # shorthand
    
    df = pd.read_csv(csv_path, sep=" ", skiprows=5, header=6)
    
    if verbose == True:  display(df.head())
    
    if unwrap == True:
            freq = np.unwrap(df['!Freq'])
    else:
        freq = df['!Freq']
    
    if plot == True:
        fig, (ax1, ax2) = plt.subplots(1, 2)

        if magn == True:
            ax1.plot(freq, df['DBS{}'.format(s)], label='Mag S{}'.format(s))
            ax2.plot(freq, df['AngS{}'.format(s)], color='orange', label='Ang S{}'.format(s))
    
        else:
            ampl = 10**(df['DBS{}'.format(s)]/20)  # convert Sxx dBm to linear
            real = np.real(ampl)
            imag = np.imag(ampl)   
            
            ax1.plot(freq, real, label='Real S{}'.format(s), color='blue') 
            ax2.plot(freq, imag, label='Imag S{}'.format(s), color='red')
            
        for ax in fig.axes:
            ax.set_xlabel('Frequency')
            ax.set_ylabel('S{}'.format(s))

            ax.set_title('Frequency vs S{}'.format(s))
            ax.legend()
            ax.grid(True)
                
        plt.suptitle('Plot of Dataset "{}"'.format(csv_path))
        plt.tight_layout()
        plt.show()

    return df


# %%
