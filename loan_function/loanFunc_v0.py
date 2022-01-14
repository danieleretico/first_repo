"""user defined functions: french loan amortization style"""

# loading packages
import numpy as np
import pandas as pd


# initialization function's inputs

S = 100.0
T = 4
i = 0.05


def loan_amortization(S, i, T):
    """


    Parameters
    ----------
    S : float
        amount borrowed.
    i : float
        Annual interest rate level.
    T : float
        Maturity.

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

    if (S < 0) | (i < 0) | (T < 0):
        raise ValueError("Inputs value must be coherent with params indications.")
    else:
        k = np.arange(0, T + 1, 1)
        k = k.T
        a_mi = (1 - (1 + i)**(-T))/i
        R = S/a_mi
        R_k = np.repeat(R, len(k))
        i_k = np.repeat(i, len(k))
        T_k = np.repeat(T, len(k))
        R_k[0] = 0
        C_k = np.zeros((len(k), 1), dtype=np.float64)
        I_k = np.zeros((len(k), 1), dtype=np.float64)
        D_k = np.zeros((len(k), 1), dtype=np.float64)
        dcf_Ck = np.zeros((len(k), 1), dtype=np.float64)
        dcf_Ik = np.zeros((len(k), 1), dtype=np.float64)
        deltaT_k = np.zeros((len(k), 1), dtype=np.float64)
        for j in range(len(k) - 1):
            if j == 0:
                C_k[0] = 0.0
                I_k[0] = 0.0
                D_k[0, 0] = S
                dcf_Ck[0] = 0
                dcf_Ik[0] = 0
                C_k[j + 1] = R*(1 + i)**(-(T-k[j]))
                I_k[j + 1] = R*(1 - (1 + i)**(-(T-k[j])))
                D_k[j + 1] = S - C_k[j + 1]
                dcf_Ck[j + 1] = (1 + i)**(-(T-k[j]))
                dcf_Ik[j + 1] = 1 - (1 + i)**(-(T-k[j]))
                deltaT_k[j + 1] = (-(T-k[j]))
            else:
                C_k[j + 1] = R*(1 + i)**(-(T-k[j]))
                I_k[j + 1] = R*(1 - (1 + i)**(-(T-k[j])))
                D_k[j + 1] = D_k[j] - C_k[j + 1]
                dcf_Ck[j + 1] = (1 + i)**(-(T-k[j]))
                dcf_Ik[j + 1] = 1 - (1 + i)**(-(T-k[j]))
                deltaT_k[j + 1] = (-(T-k[j]))
    k_index = pd.DataFrame(k)
    i_k = pd.DataFrame(i_k)
    T_k = pd.DataFrame(T_k)
    dfC_k = pd.DataFrame(R_k)
    dfC_k = pd.DataFrame(C_k)
    dfI_k = pd.DataFrame(I_k)
    dfD_k = pd.DataFrame(D_k)
    df_dcfCk = pd.DataFrame(dcf_Ck)
    df_dcfIk = pd.DataFrame(dcf_Ik)
    df_deltaT_k = pd.DataFrame(deltaT_k)
    df_loan = pd.DataFrame()
    df_loan['time k'] = k_index
    df_loan['payments'] = R_k
    df_loan['notional'] = dfC_k
    df_loan['interest'] = dfI_k
    df_loan['debt amount at k'] = D_k
    df_loan['i'] = i_k
    df_loan['delta maturity'] = deltaT_k
    df_loan['original maturity'] = T_k
    return df_loan.style.format("{:.4f}")


