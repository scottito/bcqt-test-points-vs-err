# %% use vscode's "jupyter" extension with interactive editing to use .py's as fake .ipynb's

"""
### Dynamic Fitting Python Script
The goal here is to perform the same fitting operation on a given set of data with the ability to modify: 
    subset of data used, fitting parameters, and fitting method employed. Therefore, we construct a generic dynamic_fit() 
    method that has input parameters for all of the above. In the future, we should probably try to do a more OOP approach 
    and create a "dataset" class, with generic class functions, and then that will give us the ability to create methods 
    that rely on using this "dataset" object directly. This would streamline what inputs & outputs are expected on what 
    side of the ("data" <-> "fit method" <-> "fit result") pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lmfit import Model, Parameters

# %%
def import_csv_as_pd(csv_path, magn=True, unwrap=True, plot=True, verbose=False):

    df = pd.read_csv(csv_path, sep=" ", skiprows=5)

    ##
    if verbose == True:
        df.head()

    ##
    if unwrap == True:
            freq = np.unwrap(df['!Freq'])
    else:
        freq = df['!Freq']

    ##
    if magn == True:
        ax1.plot(freq, df['DBS21'], label='Mag S21')
        ax2.plot(freq, df['AngS21'], color='orange', label='Ang S21')
    else:
        ampl = 10**(df['DBS21']/20)  # convert S21 dBm to linear
        real = np.real(ampl)
        imag = np.imag(ampl)        
        ax1.plot(freq, real, label='Real S21', color='orange')
        ax2.plot(freq, imag, label='Imag S21')

    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('S21')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('S21')
    ax1.set_title('Frequency vs S21')
    ax2.set_title('Frequency vs S21')
    ax1.legend()
    ax2.legend()
    ax1.grid(True)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    ##
    if plot == True:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(freq, real, label='Real S21', color='orange')
        ax2.plot(freq, imag, label='Imag S21')


    return df


def dynamic_fit(dataset_freqs, dataset_real, dataset_imag,  fitting_model, Q, Qc, **kwargs):

    """
    Inputs:
        dataset_freqs : [numpy array] - contains all frequencies of scan, goes low->high
        dataset_real  : [numpy array] - contains all real data points of dataset, 
                                            ordered by freq
        dataset_imag  : [numpy array] - contains all imaginary components of dataset, 
                                            ordered by freq
        fitting_model : [python method] - method that takes f0, Q, Qc, and any **kwargs, 
                                            and returns a numpy array of complex S21 vals
        
        f0 <---- removed this fit parameter, will reintroduce as an optional input later
        
        Q, Qc         : [python floats] - fitting parameters: 
                                            f0 - resonant frequency
                                            Q  - total Q
                                            Qc - coupling Q
        
        **kwargs      : [named args] - not implemented, will be used to add custom parameters 
                                            in the future once we add the ability to use 
                                            different types of fitting models


    Outputs:
        fit_result    : [lmfit model.fit_result object] - result of fitting procedure, contains
                                            information about the one dynamic_fit performed

        model_obj     : [lmfit model object] - lmfit model obj with updated values after having
                                            prepared it for fitting. Does not contain information
                                            about the fit that was performed.
        

    """
    # create model object using lmfit
    model_obj = Model(fitting_model)
    
    ######### prepare arrays for fitting
    # S21 will have a dip at resonance, to 1st order this is good appx
    resonance_idx = dataset_real.argmin() 
    f0_guess = dataset_freqs[resonance_idx]  # convert GHz to Hz
    f0 = f0_guess
    # TODO: add conversion from dB -> linear units
    # ampl = 10**(df['DBS21']/20)  # convert S21 dBm to linear

    print(f0_guess)
    
    ######### create parameter object for Lmfit
    params = Parameters()
    params.add('f0', f0, min=f0-1e2, max=f0+1e2 )  # do not want f0 to move more than a few Hz
    params.add('Q',  Q, min=1e3, max=1e9) # sensible lower/upper bounds
    params.add('Qc', Qc, min=1e3, max=1e9)
    # params['f0'].vary = False  # honestly safer to do this

    ######### perform fit, display results
    fit_result = model_obj.fit(dataset_real, params, f=dataset_freqs)

    return fit_result, model_obj



def display_dynamic_fit(fit_result, model_obj):

    f0 = fit_result.best_values["f0"]

    # TODO: implement this function, the code below was copy pasted from cell above
    ######### prepare fake data and start plot
    fig1, (ax1, ax2) = plt.subplots(2,1)
    # fig1, ax1 = plt.subplots(1)
    # ax1.plot((fake_freq - f0)/1e3, fake_vals, label='Sim Data')
    # ax1.set_title("Lmfit")

    # ax1.plot((fake_freq - f0)/1e3, result.best_fit, linestyle='--', color='red', label='Lmfit')


    # ax1.legend()

    # #########
    # residuals = fake_vals - result.best_fit
    # ax2.scatter((fake_freq - f0_guess)/1e3, residuals, label='$S_{21}^{sim} - S_{21}^{fit}$', marker='.')
    # ax2.set_title("Zoomed in Fit Residuals")
    # ax2.legend()

    # # ax2.axhline(0)

    # ######### print some values before plot is shown
    # # print("Model Parameters:")
    # # display(params)
    # print(result.fit_report())
    # ax1.set_xlabel("Detuning $\delta$ from $f_0$ = {:1.1f} GHz [kHz]".format(f0_guess/1e9))
    # ax1.set_ylabel("Re [$S_{21}$]")
    # ax2.set_xlabel("Detuning $\delta$ from $f_0$ = {:1.1f} GHz [kHz]".format(f0_guess/1e9))
    # ax2.set_ylabel("Re [$S_{21}$]")

    # # zoom_span = 0.1e6
    # # ax1.set_xlim(np.array([-zoom_span*5/1e3,   zoom_span*5/1e3]))
    # # ax2.set_xlim(np.array([-zoom_span/1e3, zoom_span/1e3]))

    # fig1.tight_layout()

    # fit_result

    return



# %%

""" now use dynamic fit on a test dataset """


dataset_one = import_csv_as_pd("IdealResonator_CAP.csv", verbose=True)