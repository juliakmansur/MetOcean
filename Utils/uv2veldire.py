import numpy as np
import sys

def uv2veldire(u,v,corr_val=[]):
    '''
    Calcula Velocidade e Direção do Vento dado [u] e [v]

    u = array/DataFrame(obrigatório)
    v = array/DataFrame (obrigatório)
    corr_value = array (opcional)
        Quando inserido um valor corrige a direção em relação a costa local.

    output:
        Velocidade do vento
        Direção do vento
    '''
    
    vel = np.sqrt(u**2+v**2)
    if corr_val != []:
        dire = np.mod(180+np.rad2deg(np.arctan2(u, v)),360) - corr_val
    else:
        dire = np.mod(180+np.rad2deg(np.arctan2(u, v)),360)
        
    return (vel,dire)

if __name__ == "__main__":
    uv2veldire()

