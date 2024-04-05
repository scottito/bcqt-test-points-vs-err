# %% use vscode's "jupyter" extension with interactive editing to use .py's as fake .ipynb's

"""
### Dynamic Fitting Python Script
The goal here is to perform the same fitting operation on a given set of data with the ability to modify: 
    subset of data used, fitting parameters, and fitting method employed. Therefore, we construct a generic dynamic_fit() 
    method that has input parameters for all of the above. In the future, we should probably try to do a more OOP approach 
    and create a "dataset" class,nump with generic class functions, and then that will give us the ability to create methods 
    that rely on using this "dataset" object directly. This would streamline what inputs & outputs are expected on what 
    side of the ("data" <-> "fit method" <-> "fit result") pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lmfit import Model, Parameters

%load_ext autoreload
%autoreload 2


# %%

def dynamic_fit(dataset_freqs, dataset_real, dataset_imag, fitting_model, Q, Qc, A, **kwargs):

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

    # print(f0_guess)
    
    ######### create parameter object for Lmfit
    params = Parameters()
    params.add('f0', f0, min=f0-50, max=f0+50 )  # do not want f0 to move more than a few Hz
    params.add('Q',  Q, min=1e3, max=1e9) # sensible lower/upper bounds
    params.add('Qc', Qc, min=1e3, max=1e9)
    params.add('A', A, min=0.1, max=10)

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






# %% attempt with fake data

from fit_models import s21_hangarmode

n_pts = 50000
Q_val = 520000
Qc_val = 90000
A_val = 1.1 

fake_freqs = np.linspace(7.91e9, 8.09e9,  n_pts)
f0_val = fake_freqs[int(n_pts/2)]

fake_s21_dB = s21_hangarmode(fake_freqs, f0_val, Q_val, Qc_val, A_val)
fake_s21_complex = 10**(fake_s21_dB/20)

fake_s21_reals = np.real(fake_s21_complex)
fake_s21_imags = np.imag(fake_s21_complex)


# plt.plot(f, fake_s21_real)

Q_guess = 520000
Qc_guess = Q_guess * 1e-1
A_guess = np.max(fake_s21_reals)

fit_result, model_obj = dynamic_fit(fake_freqs, fake_s21_reals, fake_s21_imags, s21_hangarmode, Q_guess, Qc_guess, A_guess)

Q_fit = fit_result.params['Q'].value
Q_err = (fit_result.params['Q'].stderr / Q_fit) * 100 # percentage

Qc_fit = fit_result.params['Qc'].value
Qc_err = (fit_result.params['Qc'].stderr / Qc_fit) * 100  # percentage

A_fit = fit_result.params['A'].value
A_err = (fit_result.params['A'].stderr / A_fit) * 100  # percentage


display(fit_result)
fit_result.plot_fit()

print("\n>> True Values <<")
print("   Q = {:1.2e} \n   Qc = {:1.2e} \n   fmin-fmax = [{:1.1e}, {:1.1e}] \n   $\\Delta$ = {:1.2} (==f0-fmin)"
      .format(Q_val, Qc_val, fake_freqs[0], fake_freqs[-1], f0_val-fake_freqs[0]))
print("\n>> Fit Parameters <<")
print("   Q = {:1.2e} +/- {:1.2f}% \n   Qc = {:1.2e} +/- {:1.2f}% \n   A = {:1f} +/- {:1.2f}%"
      .format(Q_fit, Q_err, Qc_fit, Qc_err, A_fit, A_err))

# %%
# TODO: create method to convert db to linear

""" now use dynamic fit on a test dataset """

from fit_models import s21_hangarmode


csv_path = "./samples/sample_data.csv"

dataset = pd.read_csv(csv_path, sep=",")
display(dataset       .head())
# plt.plot(dataset_)


# dataset_freq = dataset_one["!Freq"] 
# if dataset_freq.max() < 1e9:
    # dataset_freq *= 1e9
    
# dataset_ampl = 10**(dataset_one['DBS21']/20)
# dataset_complex = dataset_ampl * np.exp(1j * np.deg2rad(dataset_one['AngS21']))
# dataset_reals = np.real(dataset_complex)
# dataset_imags = np.imag(dataset_complex)
    
# dataset_reals

# Q_guess = 1e4
# Qc_guess = Q_guess * 1e-1

# fit_result, model_obj = dynamic_fit(dataset_one["!Freq"], dataset_reals, dataset_imags, s21_hangarmode, Q_guess, Qc_guess)

# display(fit_result)