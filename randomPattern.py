import numpy as np
import cv2
import matplotlib.image as pltimage
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import scipy
import os

Random_Pattern_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100"
Random_Pattern_URL_Absolute_Path = os.path.abspath(Random_Pattern_URL)
Random_Pattern_Data_List = os.listdir(Random_Pattern_URL_Absolute_Path)
Random_Pattern_Data_List.pop()

print(Random_Pattern_Data_List)

R_Matrix_1 = []
#increment_for_loop_1 = 0

for x in range(64):
    #x = x + increment_for_loop_1
    #increment_for_loop_1 = increment_for_loop_1 + 1
    #y = x + 1

    R_URL_1 = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100/" + Random_Pattern_Data_List[x]
    R_URL_1_Absolute_Path = os.path.abspath(R_URL_1)
    R_1 = cv2.imread(R_URL_1_Absolute_Path, cv2.IMREAD_UNCHANGED)
    R_1 = R_1.astype('float64')
    R_1_Resize = cv2.resize(R_1, (100, 100), interpolation = cv2.INTER_LINEAR)
    R_1_Vector = np.reshape(R_1_Resize, (10000, 1))
    if x == 0:
        R_Matrix_1 = R_1_Vector
    else:
        R_Matrix_1 = np.column_stack((R_Matrix_1, R_1_Vector))

#R_Matrix_2 = []

#for y in range(64, 100):
    #R_URL_2 = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100/" + Random_Pattern_Data_List[y]
    #R_URL_2_Absolute_Path = os.path.abspath(R_URL_2)
    #R_2 = cv2.imread(R_URL_2_Absolute_Path, cv2.IMREAD_UNCHANGED)
    #R_2 = R_2.astype('float64')
    #R_2_Resize = cv2.resize(R_2, (100, 100), interpolation = cv2.INTER_LINEAR)
    #R_2_Vector = np.reshape(R_2_Resize, (10000, 1))
    #if y == 64:
        #R_Matrix_2 = R_2_Vector
    #else:
        #R_Matrix_2 = np.column_stack((R_Matrix_2, R_2_Vector))

R_New_1 = np.array(R_Matrix_1)
#R_New_2 = np.array(R_Matrix_2)

print("First 64 of the random speckle patterns turned into vectors and appended together \n",  R_New_1)
print("Shape of the first combined random speckle pattern matrix \n", R_New_1.shape)

#print("Last 32 of the random speckle patterns turned into vectors and appended together \n",  R_New_2)
#print("Shape of the second combined random speckle pattern matrix \n", R_New_2.shape)

H_New_1 = hadamard(64)
print("Hadamard matrix of order 64 \n", H_New_1)
print("Shape of Hadamard matrix of order 64", H_New_1.shape)

#H_New_2 = hadamard(32)
#print("Hadamard matrix of order 32 \n", H_New_1)
#print("Shape of Hadamard matrix of order 32", H_New_1.shape)

TM_New_1 = np.matmul(R_New_1, (H_New_1.transpose()/64))
print("Transmission Matrix for the first combined speckle pattern matrix \n", TM_New_1)
print("Shape of the first Transmission matrix", TM_New_1.shape)

#TM_New_2 = np.matmul(R_New_2, (H_New_2.transpose()/32))
#print("Transmission Matrix for the second combined speckle pattern matrix \n", TM_New_2)
#print("Shape of the second Transmission matrix", TM_New_2.shape)

R_Recover_Test_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100/000001.tif"
R_Recover_Test_URL_Absolute_Path = os.path.abspath(R_Recover_Test_URL)
R_Recover_Test = cv2.imread(R_Recover_Test_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
R_Recover_Test = R_Recover_Test.astype("float64")
R_Recover_Test_Resize = cv2.resize(R_Recover_Test, (100, 100), interpolation = cv2.INTER_LINEAR)
R_Recover_Test_Vector = np.reshape(R_Recover_Test_Resize, (10000, 1))
np.seterr(invalid="ignore")
H_Recover_Test = np.matmul(np.linalg.pinv(TM_New_1), R_Recover_Test_Vector)
H_Recover_Test_Reshape = np.reshape(H_Recover_Test, (8, 8))
print("Test Image Recovery \n", H_Recover_Test_Reshape)
print("Shape of Test Image Recovery \n", H_Recover_Test_Reshape.shape)

plt.subplot(121)
plt.title("Recovered Test Image")
plt.imshow(H_Recover_Test_Reshape)
plt.show()
