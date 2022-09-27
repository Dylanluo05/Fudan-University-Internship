import numpy as np

''' Re-assign grid-ordered 16x16 pixels to round shape
    grid:
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * *
    round:
    - - - - - - - * * * * - - - - - - -
    - - - * * * * * * * * * * * * - - -
    - - - * * * * * * * * * * * * - - -
    - * * * * * * * * * * * * * * * * -
    - * * * * * * * * * * * * * * * * -
    - * * * * * * * * * * * * * * * * -
    - * * * * * * * * * * * * * * * * -
    * * * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * * * *
    * * * * * * * * * * * * * * * * * *
    - * * * * * * * * * * * * * * * * -
    - * * * * * * * * * * * * * * * * -
    - * * * * * * * * * * * * * * * * -
    - * * * * * * * * * * * * * * * * -
    - - - * * * * * * * * * * * * - - -
    - - - * * * * * * * * * * * * - - -
    - - - - - - - * * * * - - - - - - -    
'''
class Round256:
    # class variable
    dim = 16
    assignMatrix = np.zeros((16**2, 18**2))
    for i in range(4):
        for k in range(1):
            assignMatrix[0+4*k+i, 18*(k)+7+i] = 1
            assignMatrix[252+4*k+i, 18*(17+k)+7+i] = 1
    for i in range(12):
        for k in range(2):
            assignMatrix[4+12*k+i, 18*(1+k)+3+i] = 1
            assignMatrix[228+12*k+i, 18*(15+k)+3+i] = 1
    for i in range(16):
        for k in range(4):
            assignMatrix[28+16*k+i, 18*(3+k)+1+i] = 1
            assignMatrix[164+16*k+i, 18*(11+k)+1+i] = 1
    for i in range(18):
        for k in range(2*2):
            assignMatrix[92+18*k+i, 18*(7+k)+i] = 1

    @staticmethod
    def assign(data):
        return np.dot(Round256.assignMatrix.T, data)