#!/usr/bin/env python
# coding: utf-8

# # Black Litterman (BL) Function
# 
# Based on [_Sector Weighting: A Detailed Implementation of Black-Litterman_](http://kpei.github.io/bl-sector-ssif/bl.html) by Kevin Pei, Senior Portfolio Analyst for Sprott Student Investment Fund. The R code provided by this source was converted to Python code, then modified to improve flexibility and remove instances of hard coding in order to allow for use in a trading strategy.
# 
# The specific parts of this algorithm are explained in more detail within the "Black-Litterman Code Explained" files in this repository.

# # Initial Setup
# ## *Load Necessary Libraries*

# In[1]:


import numpy as np
from numpy.linalg import inv as solver
import pandas as pd
from datetime import timedelta


# ## *Define Helper Functions to be Used within the BL Function*
# 
# These functions include the impact and effect matrix generators, as well as the uncertainty matrix optimization formula to be used in the Black-Litterman function defined later.

# In[2]:


# impact_maker creates the impact matrix from the level output.
# lvl_ret is a dictionary of levels and their corresponding expected return
def impact_maker(vec, lvl_ret={0:-0.04719978, 1:-0.02474642, 2:-0.014198, 3:-0.00443905, 4:0.00456932,
                               5:0.01424941, 6:0.0241984, 7:0.04794779}):
    # Find the minimum element of the input vector
    m_vec = min(vec)
    # For each element in the input vector that is not the minimum, make a vector of the expected returns
    # corresponding to the element
    q = [lvl_ret[i] for i in vec if i!=m_vec]
    # Return the impact vector
    return(q)

# effect_maker creates the effect matrix from the level output
def effect_maker(vec):
    # Determine the minimum element of vec
    m_vec = min(vec)
    # Generate empty matrix to hold the views. Rows are the number of entries not equal to the minimum level, 
    # columns are the total number of entries in vec.
    p = np.zeros((len(vec)-vec.count(m_vec),len(vec)))
    
    # Create dummy variables ind and j to keep track of the index of vec we are looking at and the row of the
    # effect matrix we are interested in respectively
    ind=0
    j=0
    # Loop through all the elements in vec
    for elem in vec:
        # If the current element is equal to the minimum, do nothing but add 1 to ind
        if elem == m_vec:
            ind=ind+1
            continue
        # If the current element isn't equal to the minimum
        else:
            # Create a list of the differences between the current element and any element in the vector that
            # is less than it. Otherwise, let the input entry be 0. 
            cur_row = [i-elem if i<elem else 0.0 for i in vec]
            # Find the denominator of the weights for current the effect matrix row by summing the absolute
            # values of all the cur_row elements
            denom = sum(np.absolute(cur_row))
            # Divide all the cur_row elements by denom
            cur_row = [i/denom for i in cur_row]
            # Set the element in the cur_row corresponding to the current element being looked at to 1
            cur_row[ind] = 1
            # Input the row into the effect matrix
            p[j]=cur_row 
            # Increment the dummy variables
            j=j+1
            ind=ind+1
            
    # Retern the generated effect matrix
    return(np.matrix(p))

# Define objective function to be minimized for step 5
# ie. squared difference between the confidence weighting and Black-Litterman output weighting)
def omega_solve(omega, w_conf_k, lam, sigma, tau, p, pi, q):
    w_single =np.matmul(solver(lam*sigma),
                        np.matmul(solver(solver(tau*sigma) + np.matmul(p.transpose(), (1/omega)*p)), 
                        (np.matmul(solver(tau*sigma), pi) + p.transpose() * (1/omega)*q)))
    return(np.matrix.sum(np.square(np.subtract(w_conf_k, w_single))))


# # Black-Litterman Function
# 
# The main Black-Litterman function.

# In[3]:


def bl(smp_out, r, mkt_cap, risk_free):
    # Extract the date, stocks, and stock return levels from the sample lstm output
    smp_out_date = smp_out[0]
    smp_out_names = list(smp_out[1].keys())
    smp_out_lvls = list(smp_out[1].values())
    
    # Create filtered log return dataframe
    test_r = r.filter(items=smp_out_names).loc[(smp_out_date-timedelta(days=180)):smp_out_date]
    
    # Standard Deviation of Stock Returns
    sigma = np.matrix(test_r.apply(np.std).values)
    # Correlation Matrix
    rho = test_r.corr()
    # Covariance Matrix
    cov_mat = np.matrix(rho * np.matmul(sigma.transpose(), sigma))
    
    # Market Cap Data for Chosen Stocks on Given Day
    test_mc = mkt_cap.filter(items=smp_out_names).loc[smp_out_date]

    # Calculate Weights from Market Caps
    test_wts = test_mc.div(test_mc.sum(axis=0), axis=0)
    test_wts
    # Get Weekly Expected Return
    r_total = np.matrix(np.matmul(np.matrix(test_r), test_wts.transpose())).transpose()
    exp_wk_r = np.mean(r_total)*5

    # 3 Month LIBOR Rate Converted to a Daily Rate by Dividing by 90
    rf = float(risk_free.loc[smp_out_date])/(100.0*90.0)
    # Variance of Total Return
    var = np.var(r_total)
    # Determine Lambda
    lam = (exp_wk_r-rf)/var
    
    w_mkt = np.matrix(test_wts)
    pi = lam * np.matmul(np.matrix(cov_mat), np.matrix(test_wts).transpose())
    
    # Create impact and effect matrices
    Q = impact_maker(smp_out_lvls)
    P = effect_maker(smp_out_lvls)
    
    # Calibrating Uncertainty ----------------------------------------------------------------------------
    # STEP 1: Set confidence levels (HARDCODE ACCURACY OF NEURAL NET HERE)
    cf_lvls = np.repeat(0.35, len(P))

    K = P.shape[0]
    tau=0.005
    omega = np.zeros((K,K))
    for i in range(0,K):
        # STEP 2
        bl_er_100 = pi + np.matmul(np.matmul(np.matmul(tau*cov_mat, P[i,:].transpose()), 
                                             solver(np.matmul(P[i,:] * tau, cov_mat*P[i,:].transpose()))),
                                   Q[i]-P[i,:]*pi)
        w_bl_100 = np.matmul(solver(lam*cov_mat), bl_er_100)

        # STEP 3
        tilt = (w_bl_100 - w_mkt.transpose())*cf_lvls[i]

        # STEP 4
        w_cf_k = w_mkt.transpose() + tilt

        # STEP 5
        m = np.arange(0.00001,0.001,0.00001)
        d = 0
        e = np.zeros(len(m))
        ind = 0
        min_e = 1000
        for j in m:
            e[d] = omega_solve(j, w_cf_k, lam, cov_mat, tau, P[i,:], pi, Q[i])       
            d=d+1

        omega[i,i] = m[np.where(e==min(e))]
        
        # Determine the Expected Returns from Black-Litterman using the Newly-Found Omega
    bl_er=np.matmul(solver(solver(tau*cov_mat) + np.matmul(P.transpose(), np.matmul(solver(omega), P))), 
                          (np.matmul(solver(tau*cov_mat), pi) + np.matmul(P.transpose(), np.matmul(solver(omega), Q)).transpose()))
    # Determine the Black-Litterman Weights from the Newly-Found Black-Litterman Expected Returns
    bl_w = np.matmul(solver(lam*cov_mat), bl_er)
    
    # Return the Black-Litterman Weights found
    return(bl_w)


# # Sample Run

# ## *Assumed Input Dataframes*

# In[4]:


# Import all log return data
in_r = pd.read_csv('log_ret_data_final.csv',index_col='date', parse_dates=True)
# Import market cap data
in_mkt_cap = pd.read_csv('mark_cap_data_final.csv',index_col='date', parse_dates=True)
# Read in and format 3 month LIBOR rate data
in_risk_free = pd.read_csv('libor_rates_final.csv',index_col='date', parse_dates=True)


# ## *Stock Forecast Output*

# In[5]:


# Sample stock forecasting algorithm output
smp_out = (pd.to_datetime('1995-07-03'),{'91704710':7, '15678210':0, '15102010':5, '53279110':5})


# ## *Run Black-Litterman Function*

# In[6]:


w = bl(smp_out, in_r, in_mkt_cap, in_risk_free)


# ## *Check Black-Litterman Output*

# In[7]:


w


# In[8]:


sum(w)

