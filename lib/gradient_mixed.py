
from . import mixmem_aux as aux
import time
import numpy as np

def gradient_descent(n,r0 = None,c0=None, wr=1E-5, wc=1E-5,
                     N=2500, verbose_time = False,
                     progressBar=False, exp = np.exp, nroles = None):
    ''' 
    Optimize a mixed membership model using gradient descent.

    Parameters:

    -  n: Real number of trips between stations, indexed as [from, to, time_bin]
    - r0: Role strength initial guess, indexed as [role_1,role_2,time]
    - c0: Station role initial guess, indexed as [station, role]
    - wc: Initial guess for the gradient weight for c0
    - rc: Initial guess for the gradient weight for r0
    -  N: Number of iterations -- TODO: Add a tolerance stop criterion

    - verbose: Whether the method should be verbose
    - progressBar: Whether a progressbar should be displayed
    
    - exp:  vectorized function to substitute the exponential in the logarithmic 
            gradient descent. Use at your own risk. The minimal requeriments on
            the function are:

            * f is positive
            * f(0) = 1
            * f is monotonous increasing, and strictly monotonous near 0.

            Numerical experiments indicate x -> (x^2+1)^1/2+x is a good function,
            but twice the inverse logit looks reasonable and way more robust. 
    - nroles: Number of roles. Only necessary if either r0 or c0 are not provided.
    
    - Returns the updated values of the parameters:
    
        (rfit,cfit, wrfit, wcfit)
        This allows for the following code:
        
        x = gradient_descent(100, r,c,wr,wc, **params)
        while (stuff(x)):
            x = gradient_descent(100,*x, **params)
            
        so that you can code your own stopping conditions
    '''
    
    #Set up the progress bar with ipywidgets
    if progressBar:
        import ipywidgets as widgets
        from IPython.display import display
        f = widgets.FloatProgress(min=0, max=N)
        display(f)
        
    #random initialisation
    
    if r0 is None:
        assert (nroles is not None)
        ntimes = n.shape[2]
        r0 = np.random.lognormal(size=(nroles,nroles,ntimes))
    if c0 is None:
        assert (nroles is not None)
        nstations = n.shape[0]
        c0 = np.random.lognormal(size=(nstations,nroles))
        
    #Compute the original likelihood
    old_l=aux.likelihood(c0,r0,n);
    
    #Factors with which we will update the weights
    #so that the speed of the Gradient Descent is Optimal
    up = 1.5
    down = 0.75
    

    if verbose_time!=False: #print first line for the table
        start_time = time.time()
        print("N: \t C_RelChange \t R_RelChange \t log_like \t d_log_like \t time(s)")
    for i in range(N):
        
        
        #Single gradient descent for the R
        #=================================
        
        rs = aux.r_score(c0,r0,n) #compute r gradient (from the mixmem_aux library)
        #Compute two new points with different log-gradient weights
        r1 = r0*exp(down*wr*rs);        r2 = r0*exp(up*wr*rs)
        r1 = np.maximum(r1, 1E-20);       r2 = np.maximum(r2, 1E-20);    
        l1 = aux.likelihood(c0,r1,n);   l2 = aux.likelihood(c0,r2,n)

        #choose (and keep) the best weight (if possible)
        if l2>l1: r0=r2;  wr*=up;    l=l2;
        else:     r0=r1;  wr*=down;  l=l1;
        
       #########################
    
        #Copy of the code above for the C
        #================================
        
        cs = aux.c_score(c0,r0,n)
                 
        c1 = c0*exp(down*wc*cs);          c2 = c0*exp(up*wc*cs)
        c1 = np.maximum(c1, 1E-20);       c2 = np.maximum(c2, 1E-20); 
        l1 = aux.likelihood(c1,r0,n);     l2 = aux.likelihood(c2,r0,n)
        
        if l2>l1:  c0=c2;  wc*=up;  l=l2;
        else:      c0=c1;  wc*=down;  l=l1;
            
        #The set of minima is not unique: we can divide c by a constant
        #and multiply r by a constant and obtain the same model. We must
        #choose an (arbitrary) normalisation. See below.
        r0,c0 = normalize(r0,c0)
            
            
        #######################
        # If the user wants information to be printed every "verbose_time" steps, do so.
        if verbose_time!=False and i%verbose_time==0:
            deltar = np.mean(np.abs(exp(up*wr*rs)-1)*r0)/np.mean(r0)
            deltac = np.mean(np.abs(exp(up*wc*cs)-1)*c0)/np.mean(c0)
            print(("{}"+(4*"\t {:.3e}")+"\t {:.2f}")
                  .format(i,deltac, deltar,l, l-old_l, time.time()-start_time))
            
        #Update likelihood
        old_l = l;         
        if progressBar:
            f.value=i+1;
    #end the progress bar    
    if progressBar:
        f.close()
    return normalize(r0,c0)+(wr,wc);


# Auxiliary normalizing function. Normalized so that the
# station roles sum to 1. 
def normalize(r,c):
    cnorm = np.sum(c,axis=0)
    rnorm = np.outer(cnorm,cnorm)
    cn = c/cnorm[np.newaxis,:]
    rn = r*rnorm[:,:,np.newaxis]
    return rn,cn
