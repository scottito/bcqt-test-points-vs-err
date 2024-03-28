# fit_models.py

## serves as a convenient place to store all fitting models to avoid cluttering other scripts
## utilize in your fitting code by adding an import statement like so:
##      from fit_models import s21_hangarmode, s21_reflection, s11

import numpy as np

def s21_hangarmode(f, f0, Q, Qc):

    """ S21 Hangarmode Model
    
    returns 3 item list with the three
    elements being:
        [0] freq array - numpy array of floats in Hz, not GHz
        [1] real components - numpy array of floats
        [2] imag components - numpy array of floats
    """


    if np.max(f) < 1e9:
        f = f * 1e9
        print("\n >>>> S21_Hangarmode Model: Error, max of freq array is less than 1e9: ({:1.4e}) \n".format(np.max(f)))
        print("   >>>>>> Converting frequency array to Hz from GHz. \n")

    s21 = 1 - (Q/Qc)/(1 + 2j*Q*(f/f0 - 1)) 

    return [f, np.real(s21), np.imag(s21)]

def s21_reflection(f, f0, Q, Qc):
    return

def s11(f, f0, Q, Qc):
    return