import numpy as np

def inertP(lat):
    '''
    Calcula o Periodo Inercial a partir da Latitude (°)

    lat = array/DataFrame(obrigatorio)

    output:
        Periodo Inercial (horas)
    '''
    lat = lat
    inertP = ((2*np.pi)/(2*0.00007292*np.sin(np.deg2rad(lat))))/3600
    
    return(inertP)

if __name__ == "__main__":
    import sys
    inertP()
