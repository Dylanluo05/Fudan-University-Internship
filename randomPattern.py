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

R_Matrix = []

for x in range(100):
    R_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100/" + Random_Pattern_Data_List[x]
    R_URL_Absolute_Path = os.path.abspath(R_URL)
    R = cv2.imread(R_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    R = R.astype('float64')
    R_Resize = cv2.resize(R, (100, 100), interpolation = cv2.INTER_LINEAR)
    R_Vector = np.reshape(R_Resize, (10000, 1))
    if x == 0:
        R_Matrix = R_Vector
    else:
        R_Matrix = np.column_stack((R_Matrix, R_Vector))

R_New = np.array(R_Matrix)

print("All of the random speckle patterns turned into vectors and appended together \n",  R_New)
print("Shape of the combined random speckle pattern matrix \n", R_New.shape)

Random_Pattern_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100/randomPattern100.npy"
Random_Pattern_URL_Absolute_Path = os.path.abspath(Random_Pattern_URL)
Random_Pattern = np.load(Random_Pattern_URL_Absolute_Path)
print("randomPattern100.npy Matrix: \n", Random_Pattern)
print("Shape of randomPattern100.npy Matrix \n", Random_Pattern.shape)

R_New_Update = np.matmul(R_New, Random_Pattern)

H_New = hadamard(256)
print("Hadamard matrix of order 256 \n", H_New)
print("Shape of Hadamard matrix of order 256", H_New.shape)

TM_New = np.matmul(R_New_Update, (H_New.transpose()/256))
print("Transmission Matrix for the first combined speckle pattern matrix \n", TM_New)
print("Shape of the first Transmission matrix", TM_New.shape)


R_Recover_Test_1_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100/000003.tif"
R_Recover_Test_1_URL_Absolute_Path = os.path.abspath(R_Recover_Test_1_URL)
R_Recover_Test_1 = cv2.imread(R_Recover_Test_1_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
R_Recover_Test_1 = R_Recover_Test_1.astype("float64")
R_Recover_Test_1_Resize = cv2.resize(R_Recover_Test_1, (100, 100), interpolation = cv2.INTER_LINEAR)
R_Recover_Test_1_Vector = np.reshape(R_Recover_Test_1_Resize, (10000, 1))
np.seterr(invalid="ignore")
H_Recover_Test_1 = np.matmul(np.linalg.pinv(TM_New), R_Recover_Test_1_Vector)
H_Recover_Test_1_Reshape = np.reshape(H_Recover_Test_1, (16, 16))

H_Recover_Test_1_Sum = 0
H_Recover_Test_1_Count = 0
for i in range(len(H_Recover_Test_1_Reshape)):
    for j in range(len(H_Recover_Test_1_Reshape[i])):
        H_Recover_Test_1_Sum += float(H_Recover_Test_1_Reshape[i][j])
        H_Recover_Test_1_Count += 1
H_Recover_Test_1_Average = float(H_Recover_Test_1_Sum)/float(H_Recover_Test_1_Count)

H_Recover_Test_1_Reshape_Threshed = H_Recover_Test_1_Reshape.copy()
H_Recover_Test_1_Reshape_Threshed[H_Recover_Test_1_Reshape_Threshed >= 4.251965770569743e-05] = 1
H_Recover_Test_1_Reshape_Threshed[H_Recover_Test_1_Reshape_Threshed < 4.251965770569743e-05] = 0
print("Test Image Recovery 1 without Thresh \n", H_Recover_Test_1_Reshape)
print("Shape of Test Image Recovery without Thresh \n", H_Recover_Test_1_Reshape.shape)
print("Test Image Recovery 1 with Thresh \n", H_Recover_Test_1_Reshape_Threshed)
print("Shape of Test Image Recovery with Thresh \n", H_Recover_Test_1_Reshape_Threshed.shape)
print("Average Value in the Test Image Recovery 1 \n", H_Recover_Test_1_Average)

R_Recover_Test_2_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100/000004.tif"
R_Recover_Test_2_URL_Absolute_Path = os.path.abspath(R_Recover_Test_2_URL)
R_Recover_Test_2 = cv2.imread(R_Recover_Test_2_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
R_Recover_Test_2 = R_Recover_Test_2.astype("float64")
R_Recover_Test_2_Resize = cv2.resize(R_Recover_Test_2, (100, 100), interpolation = cv2.INTER_LINEAR)
R_Recover_Test_2_Vector = np.reshape(R_Recover_Test_2_Resize, (10000, 1))
np.seterr(invalid="ignore")
H_Recover_Test_2 = np.matmul(np.linalg.pinv(R_New_Update), R_Recover_Test_2_Vector)
H_Recover_Test_2_Reshape = np.reshape(H_Recover_Test_2, (16, 16))

H_Recover_Test_2_Sum = 0
H_Recover_Test_2_Count = 0
for i in range(len(H_Recover_Test_2_Reshape)):
    for j in range(len(H_Recover_Test_2_Reshape[i])):
        H_Recover_Test_2_Sum += float(H_Recover_Test_2_Reshape[i][j])
        H_Recover_Test_2_Count += 1
H_Recover_Test_2_Average = float(H_Recover_Test_2_Sum)/float(H_Recover_Test_2_Count)

H_Recover_Test_2_Reshape_Threshed = H_Recover_Test_2_Reshape.copy()
H_Recover_Test_2_Reshape_Threshed[H_Recover_Test_2_Reshape_Threshed >= 4.251965770569743e-05] = 1
H_Recover_Test_2_Reshape_Threshed[H_Recover_Test_2_Reshape_Threshed < 4.251965770569743e-05] = 0
print("Test Image Recovery 2 without Thresh \n", H_Recover_Test_2_Reshape)
print("Shape of Test Image Recovery 2 without Thresh \n", H_Recover_Test_2_Reshape.shape)
print("Test Image Recovery 2 with Thresh \n", H_Recover_Test_2_Reshape_Threshed)
print("Shape of Test Image Recovery 2 with Thresh \n", H_Recover_Test_2_Reshape_Threshed.shape)
print("Average Value in the Test Image Recovery 2 \n", H_Recover_Test_2_Average)

plt.subplot(121)
plt.title("Recovered Test Image 1 \n without Thresh")
plt.imshow(H_Recover_Test_1_Reshape)
plt.subplot(122)
plt.title("Recovered Test Image 1\n with Thresh")
plt.imshow(H_Recover_Test_1_Reshape_Threshed)
plt.show()

plt.subplot(121)
plt.title("Recovered Test Image 2 \n without Thresh")
plt.imshow(H_Recover_Test_2_Reshape)
plt.subplot(122)
plt.title("Recovered Test Image 2 \n with Thresh")
plt.imshow(H_Recover_Test_2_Reshape_Threshed)
plt.show()


H_1_Matrix = []

for k in range(100):
    R_1_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100/" + Random_Pattern_Data_List[k]
    R_1_URL_Absolute_Path = os.path.abspath(R_1_URL)
    R_1 = cv2.imread(R_1_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    R_1 = R_1.astype("float64")
    R_1_Resize = cv2.resize(R_1, (100, 100), interpolation = cv2.INTER_LINEAR)
    R_1_Vector = np.reshape(R_1_Resize, (10000, 1))
    np.seterr(invalid="ignore")
    H_1 = np.matmul(np.linalg.pinv(TM_New), R_1_Vector)
    H_1_Vector = np.reshape(H_1, (256, 1))
    if k == 0:
        H_1_Matrix = H_1_Vector
    else:
        H_1_Matrix = np.column_stack((H_1_Matrix, H_1_Vector))

H_1_Matrix_New = np.array(H_1_Matrix)
print("Combined Test Matrix \n", H_1_Matrix_New)
print("Shape of Test Combined Test Matrix", H_1_Matrix_New.shape)

H_1_Matrix_New_Edit = np.matmul(H_1_Matrix_New, Random_Pattern)

Save_H_1_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/Recovered Random Patterns/Combined Test Image.tif"
Save_H_1_URL_Absolute_Path = os.path.abspath(Save_H_1_URL)

plt.subplot(121)
plt.title("Test Combined Matrix")
plt.imshow(H_1_Matrix_New)
plt.subplot(122)
plt.title("Edited Test Combined Matrix")
plt.imshow(H_1_Matrix_New_Edit)
plt.savefig(Save_H_1_URL_Absolute_Path)

H_Recover_Average_List = []

for w in range(100):
    R_Recover_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100/" + Random_Pattern_Data_List[w]
    R_Recover_URL_Absolute_Path = os.path.abspath(R_Recover_URL)
    R_Recover = cv2.imread(R_Recover_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    R_Recover = R_Recover.astype("float64")
    R_Recover_Resize = cv2.resize(R_Recover, (100, 100), interpolation = cv2.INTER_LINEAR)
    R_Recover_Vector = np.reshape(R_Recover_Resize, (10000, 1))
    np.seterr(invalid="ignore")
    H_Recover = np.matmul(np.linalg.pinv(TM_New), R_Recover_Vector)
    H_Recover_Reshape = np.reshape(H_Recover, (16, 16))

    H_Recover_Sum = 0
    H_Recover_Count = 0
    for u in range(len(H_Recover_Reshape)):
        for v in range(len(H_Recover_Reshape[u])):
            H_Recover_Sum += float(H_Recover_Reshape[u][v])
            H_Recover_Count += 1
    H_Recover_Average = float(H_Recover_Sum)/float(H_Recover_Count)
    H_Recover_Average_List.append(H_Recover_Average)

    H_Recover_Reshape_Threshed = H_Recover_Reshape.copy()
    H_Recover_Reshape_Threshed[H_Recover_Reshape_Threshed >= 4.251965770569743e-05] = 1
    H_Recover_Reshape_Threshed[H_Recover_Reshape_Threshed < 4.251965770569743e-05] = 0

    Save_H_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/Recovered Random Patterns/" + Random_Pattern_Data_List[w]
    Save_H_URL_Absolute_Path = os.path.abspath(Save_H_URL)

    plt.subplot(121)
    plt.title("Recovered Random Pattern without Thresh")
    plt.imshow(H_Recover_Reshape)
    plt.subplot(122)
    plt.title("Recovered Random Pattern with Thresh")
    plt.imshow(H_Recover_Reshape_Threshed)
    plt.savefig(Save_H_URL_Absolute_Path)


H_Recover_Average_Overall_Sum = 0
for a in range(len(H_Recover_Average_List)):
    H_Recover_Average_Overall_Sum += float(H_Recover_Average_List[a])
H_Recover_Average_Overall = float(H_Recover_Average_Overall_Sum)/float(len(H_Recover_Average_List))

Save_Average_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/Recovered Random Patterns/Random Patterns Average Values.txt"
Save_Average_URL_Absolute_Path = os.path.abspath(Save_Average_URL)
Average_Write = open(Save_Average_URL_Absolute_Path, "a")
Average_Write.write(str(H_Recover_Average_List))
Average_Write.write("\n \n" + str(H_Recover_Average_Overall))
Average_Write.close()






