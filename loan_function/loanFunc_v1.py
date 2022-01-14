"""user defined functions: french loan amortization style version 1."""

# loading packages
import numpy as np
import pandas as pd


S = 100.0
T = 4
i = 0.05
freq = 6

def loan_amortizationVectorized(S, i, T, freq):
    """


    Parameters
    ----------
    S : float
        amount borrowed.
    i : float
        Annual interest rate level.
    T : float
        Maturity.
    Freq : float
        Notional payment frequency: 1: Annual
                                    12 Monthly
                                    4  Quaterly
                                    6  Semiannual
    preAmm : int
            number of periods since amortization occurs starting from t = 0

    Raises
    ------
    ValueError
        Inputs value must be coherent with params indications.

    Returns
    -------
    df_loan : DataFrame
        returns a DataFrame object.

    """
    import numpy as np
    import pandas as pd

    if (S < 0) | (i < 0) | (T < 0) | (freq <= 0):
        raise ValueError("Inputs value must be coherent with params indications")
    else:
        im = (1 + i)**(1/freq) - 1   # interest rate compounding equivalency
        a_freqmi = (1 - (1 + i)**(-T))/im  # annuity calculation
        R = S/a_freqmi  # amortizing amount payid every period

        # initializing times vector

        k = -np.arange(freq * T + 1, 0, -1)
        k[0] = 0  # debt is repaid at the end of the period

        # initialization and calculation of discount factors
        dcf_kP = np.zeros((len(k), ), dtype=np.float64) + 1 + im
        dcf_kP = np.power(dcf_kP, k.T)
        dcf_kP[0] = 0

        # initializing and calculating notional and interest payment, debt after notional payment respectively (C_k, I_k, D_k)
        # vectors initialization
        C_k = np.zeros((len(k), ), dtype=np.float64)
        I_k = np.zeros((len(k), ), dtype=np.float64)
        D_k = np.zeros((len(k), ), dtype=np.float64)
        C_k = R * dcf_kP  # notional payments vector

        I_k = R * (1 - dcf_kP)  # interest payments vector
        I_k[0] = 0  # no interest payment at inception

        D_k = np.repeat(S, len(k)) - (np.cumsum(C_k))  # debt amount before notional payments vector

        # initialazing DataFrame object to store produced vectors

        dfC_k = pd.DataFrame(C_k)
        dfI_k = pd.DataFrame(I_k)
        dfD_k = pd.DataFrame(D_k)
        df_loan = pd.DataFrame()
        df_loan['notional'] = dfC_k
        df_loan['interest'] = dfI_k
        df_loan['payments'] = dfC_k + dfI_k
        df_loan['debt amount at k'] = dfD_k
        return df_loan.style.format("{:.4f}")

