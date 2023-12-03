import numpy as np

def veldire2uv(vel,dire,par='wnd'):
    '''
    Calcula [u] e [v] dado Velocidade e Direção 

    vel = array/DataFrame(obrigatorio)
    dire = array/DataFrame (obrigatorio)
    par = string (opcional default 'wnd')
        defina o parametro que esta sendo analisado

    Detalhes :
        parametros - escolha dependendo da analise
                wnd = wind
                curr = current

    output:
        u, v 
    '''
    a = np.deg2rad(dire)
    if par=='wnd':    
        u = -vel*np.sin(a)
        v = -vel*np.cos(a)
    else:
        u = vel*np.sin(a)
        v = vel*np.cos(a)

    return (u,v)