import sys
import numpy as np

def rmse(predictions, targets):
    ''' Calula a Raiz do erro Médio Quadrado (Root Mean Squared Error) de um valor em relação ao outro.
    prediction - Array/DataFrame (Obrigatório)
    targets - Array/DataFrame (Obrigatório)
    '''

    err = np.sqrt(((predictions - targets) ** 2).mean())
    return (err)

if __name__ == "__main__":
    import sys
    rmse()
