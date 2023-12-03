import numpy as np

def compute_histogram(variable, bin_max, bin_min, dbin, pdf=False):
    """ Computes 1D histogram or probability density for a given variable.
        
    Keyword arguments:
    variable -- 1D array.
    bin_max -- maximum value for bins
    bin_min -- minimum value for bins
    dbin -- bin size
    pdf -- (default False)
    
    Returns:
    bins -- histogram bins
    counts -- either counts or probability density
        
    """
    bins = np.arange(bin_min, bin_max, dbin)
    count = []
    for i in range(len(bins)):
        ind = (variable>bins[i] - dbin/2) & (variable<=bins[i]+dbin/2)
        count.append(ind.sum())
    count = np.array(count)
    if pdf:
        norm_hist = count/count.sum()/dbin
        assert np.allclose(norm_hist.sum()*dbin, 1.0), "PDF doesn't sum to 1"
    
        return(bins, norm_hist)
    else:
        return(bins, count)

if __name__ == "__main__":
    import sys
    compute_histogram()