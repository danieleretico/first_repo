"""user defined functions: french loan amortization style version 2"""

import numpy as np
import pandas as pd

S = 100.0
T = 10
i = 0.05
freq = 1
preAmm = 0


def loan_amortizationVectorizedV2(S, i, T, freq, preAmm):
    """


    Parameters
    ----------
    S : float
        amount borrowed.
    i : float
        Annual interest rate.
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


    if (S < 0) | (i < 0) | (T < 0) | (freq <= 0) | (preAmm < 0):  # input params control
        raise ValueError("Inputs value must be coherent with params indications")
    else:
        if preAmm == 0:  # Case: Amortization not delayed
            im = (1 + i)**(1/freq) - 1   # interest rate compounding equivalency
            a_freqmi = (1 - (1 + i)**(-T))/im  # annuity calculation
            R = S/a_freqmi  # amortizing amount payid every period

            # initializing times vector

            k = -np.arange(freq * T + 1, 0, -1) 
            k[0] = 0  # debt is repaid at the end of the period, a t = 0 no notional payments

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
            I_k[0] = 0  # no interest payment t = 0

            D_k = np.repeat(S, len(k)) - (np.cumsum(C_k))  # debt amount before notional payments vector

            # initialazing DataFrame object to store results

            dfC_k = pd.DataFrame(C_k)
            dfI_k = pd.DataFrame(I_k)
            dfD_k = pd.DataFrame(D_k)
            df_loan = pd.DataFrame()
            df_loan['Notional'] = dfC_k
            df_loan['Interest'] = dfI_k
            df_loan['Payment notional/intererest'] = dfC_k + dfI_k
            df_loan['Debt amount'] = dfD_k
            return df_loan.style.format("{:.4f}")

        else:
            k = -np.arange(freq * T + 1, 0, -1)
            if len(k) - preAmm == 1:  # bullet loan condition
                R_k = np.zeros((len(k), ), dtype=np.float64)
                R_k[1:len(k)] = S*i
                I_k = R_k
                C_k = np.zeros((len(k), ), dtype=np.float64)
                C_k[len(k) - 1] = S
                D_k = np.repeat(S, len(k))
                D_k[len(k) - 1] = 0

                dfC_k = pd.DataFrame(C_k)
                dfI_k = pd.DataFrame(I_k)
                dfD_k = pd.DataFrame(D_k)
                df_loan = pd.DataFrame()
                df_loan['Notional'] = dfC_k
                df_loan['Interest'] = dfI_k
                df_loan['Payment notional/intererest'] = dfC_k + dfI_k
                df_loan['Debt amount'] = dfD_k
                return df_loan.style.format("{:.4f}")
            else:  # all other types of loans with preAmm numbers of delayed payment
                k = -np.arange(freq * T + 1, 0, -1)
                im = (1 + i)**(1/freq) - 1   # interest rate compounding equivalency
                a_freqmi = (1 - (1 + i)**(- (T - preAmm)))/im  # annuity calculation
            
                R_preAmm = np.zeros((len(k), ), dtype=np.float64)
                R_preAmm[(preAmm + 1):len(k)] = S/a_freqmi  # amortizing amount paid every period

                dcf_kP = np.zeros((len(k), ), dtype=np.float64) + 1 + im
                dcf_kP = np.power(dcf_kP, k.T)
                dcf_kP[0:(preAmm + 1)] = 0
            
                C_preAmm = np.zeros((len(k), ), dtype=np.float64)
                C_preAmm = R_preAmm * dcf_kP

                # R_preAmm[1:(preAmm + 1)] = S*i
                I_preAmm = np.zeros((len(k), ), dtype=np.float64)
                I_preAmm = R_preAmm * (1 - dcf_kP)
                I_preAmm[1:(preAmm + 1)] = S*im
            
                D_preAmm = np.repeat(S, len(k)) - (np.cumsum(C_preAmm))

                dfC_preAmm = pd.DataFrame(C_preAmm)
                dfI_preAmm = pd.DataFrame(I_preAmm)
                dfD_preAmm = pd.DataFrame(D_preAmm)
                df_loan = pd.DataFrame()
                df_loan['Notional'] = dfC_preAmm
                df_loan['Interest'] = dfI_preAmm
                df_loan['Payment notional/intererest'] = dfC_preAmm + dfI_preAmm
                df_loan['Debt amount'] = dfD_preAmm
                return df_loan.style.format("{:.4f}")



