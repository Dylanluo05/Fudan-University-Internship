import numpy as np
import cv2
import matplotlib.image as pltimage
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import scipy
import os

#Load all speckle patterns, turn them into vectors
#Append them all together to create a giant matrix with dimensions 10000 rows x 256 columns

S_Matrix = []
increment_for_loop_1 = 0

MMF_Data_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/calibdata/"
MMF_Data_URL_Absolute_Path = os.path.abspath(MMF_Data_URL)
MMF_Data_List = os.listdir(MMF_Data_URL_Absolute_Path)
print("MMF List Data \n", MMF_Data_List)

for x in range(256):
    x = x + increment_for_loop_1
    increment_for_loop_1 = increment_for_loop_1 + 1
    y = x + 1
    S_Negative_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/calibdata/" + MMF_Data_List[x]
    S_Negative_URL_Absolute_Path = os.path.abspath(S_Negative_URL)
    S_Negative = cv2.imread(S_Negative_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    S_Negative = S_Negative.astype("float64")

    S_Positive_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/calibdata/" + MMF_Data_List[y]
    S_Positive_URL_Absolute_Path = os.path.abspath(S_Positive_URL)
    S_Positive = cv2.imread(S_Positive_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    S_Positive = S_Positive.astype("float64")

    S_Difference = np.subtract(S_Positive, S_Negative)
    S_Resize = cv2.resize(S_Difference, (100, 100), interpolation = cv2.INTER_LINEAR)
    S_Vector = np.reshape(S_Resize, (10000, 1))
    if x == 0:
        S_Matrix = S_Vector
    else:
        S_Matrix = np.column_stack((S_Matrix, S_Vector))

S_New = np.array(S_Matrix)
print("All of the Speckle Patterns Turned into Vectors and Appended Together \n", S_New)
print("Shape of All of the Speckle Patterns Appended Together \n", S_New.shape)

#Generating a Hadamard Matrix

H_New = hadamard(256)

print("Hadamard Matrix of Order 256 \n", H_New)
print("Shape of Order 256 Hadamard Matrix \n", H_New.shape)

#Finding the Transmission Matrix

# H_New.transpose()/16 is the same as np.linalg.inv(H_New).
# This is why Hadamard Matrices are useful for finding the Transmission Matrix
TM = np.matmul(S_New, (H_New.transpose()/256))
print("Transmission Matrix \n", TM)
print("Shape of Transmission Matrix \n", TM.shape)

Random_Pattern_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100"
Random_Pattern_URL_Absolute_Path = os.path.abspath(Random_Pattern_URL)
Random_Pattern_Data_List = os.listdir(Random_Pattern_URL_Absolute_Path)
Random_Pattern_Data_List.pop()

print(Random_Pattern_Data_List)

Random_Pattern_100_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100/randomPattern100.npy"
Random_Pattern_100_URL_Absolute_Path = os.path.abspath(Random_Pattern_100_URL)
Random_Pattern_100 = np.load(Random_Pattern_100_URL_Absolute_Path)
print("randomPattern100.npy Matrix: \n", Random_Pattern_100)
print("Shape of randomPattern100.npy Matrix \n", Random_Pattern_100.shape)

R_Recover_Test_1_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100/000007.tif"
R_Recover_Test_1_URL_Absolute_Path = os.path.abspath(R_Recover_Test_1_URL)
R_Recover_Test_1 = cv2.imread(R_Recover_Test_1_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
R_Recover_Test_1 = R_Recover_Test_1.astype("float64")
R_Recover_Test_1_Resize = cv2.resize(R_Recover_Test_1, (100, 100), interpolation = cv2.INTER_LINEAR)
R_Recover_Test_1_Vector = np.reshape(R_Recover_Test_1_Resize, (10000, 1))
np.seterr(invalid="ignore")
H_Recover_Test_1 = np.matmul(np.linalg.pinv(TM), R_Recover_Test_1_Vector)
H_Recover_Test_1_Reshape = np.reshape(H_Recover_Test_1, (16, 16))
Random_Pattern_Correspondant_Test_1 = Random_Pattern_100[6]
Random_Pattern_Correspondant_Test_1_Reshape = np.reshape(Random_Pattern_Correspondant_Test_1, (16, 16))
plt.subplot(121)
plt.title("Random Pattern Image Recovery \n Test 1")
plt.imshow(H_Recover_Test_1_Reshape)
plt.subplot(122)
plt.title("Random Pattern Image Recovery \n Test 1 Ideal Random Pattern \n Counterpart")
plt.imshow(Random_Pattern_Correspondant_Test_1_Reshape)
plt.show()

R_Recover_Test_2_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100/000008.tif"
R_Recover_Test_2_URL_Absolute_Path = os.path.abspath(R_Recover_Test_2_URL)
R_Recover_Test_2 = cv2.imread(R_Recover_Test_2_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
R_Recover_Test_2 = R_Recover_Test_2.astype("float64")
R_Recover_Test_2_Resize = cv2.resize(R_Recover_Test_2, (100, 100), interpolation = cv2.INTER_LINEAR)
R_Recover_Test_2_Vector = np.reshape(R_Recover_Test_2_Resize, (10000, 1))
np.seterr(invalid="ignore")
H_Recover_Test_2 = np.matmul(np.linalg.pinv(TM), R_Recover_Test_2_Vector)
H_Recover_Test_2_Reshape = np.reshape(H_Recover_Test_2, (16, 16))
Random_Pattern_Correspondant_Test_2 = Random_Pattern_100[7]
Random_Pattern_Correspondant_Test_2_Reshape = np.reshape(Random_Pattern_Correspondant_Test_2, (16, 16))
plt.subplot(121)
plt.title("Random Pattern Image Recovery \n Test 2")
plt.imshow(H_Recover_Test_2_Reshape)
plt.subplot(122)
plt.title("Random Pattern Image Recovery \n Test 2 Ideal Random Pattern \n Counterpart")
plt.imshow(Random_Pattern_Correspondant_Test_2_Reshape)
plt.show()


H_Recover_Average_List = []
H_Recover_Median_Full_List = []

for w in range(100):
    R_Recover_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/randomPattern100/" + Random_Pattern_Data_List[w]
    R_Recover_URL_Absolute_Path = os.path.abspath(R_Recover_URL)
    R_Recover = cv2.imread(R_Recover_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    R_Recover = R_Recover.astype("float64")
    R_Recover_Resize = cv2.resize(R_Recover, (100, 100), interpolation = cv2.INTER_LINEAR)
    R_Recover_Vector = np.reshape(R_Recover_Resize, (10000, 1))
    np.seterr(invalid="ignore")
    H_Recover = np.matmul(np.linalg.pinv(TM), R_Recover_Vector)
    H_Recover_Reshape = np.reshape(H_Recover, (16, 16))

    H_Recover_Sum = 0
    H_Recover_Count = 0
    H_Recover_Median_List = []
    for u in range(len(H_Recover_Reshape)):
        for v in range(len(H_Recover_Reshape[u])):
            H_Recover_Sum += float(H_Recover_Reshape[u][v])
            H_Recover_Count += 1
            H_Recover_Median_List.append(float(H_Recover_Reshape[u][v]))
    H_Recover_Average = float(H_Recover_Sum)/float(H_Recover_Count)
    H_Recover_Average_List.append(H_Recover_Average)

    H_Recover_Median_List_Sorted = sorted(H_Recover_Median_List)
    H_Recover_Median = (float(H_Recover_Median_List_Sorted[128]) + float(H_Recover_Median_List_Sorted[129]))/float(2)
    H_Recover_Median_Full_List.append(H_Recover_Median)

    H_Recover_Reshape_Threshed = H_Recover_Reshape.copy()
    H_Recover_Reshape_Threshed[H_Recover_Reshape_Threshed >= 0.5] = 1
    H_Recover_Reshape_Threshed[H_Recover_Reshape_Threshed < 0.5] = 0

    Random_Pattern_Correspondant = Random_Pattern_100[w]
    Random_Pattern_Correspondant_Reshape = np.reshape(Random_Pattern_Correspondant, (16, 16))

    Save_H_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/Recovered Random Patterns/" + Random_Pattern_Data_List[w]
    Save_H_URL_Absolute_Path = os.path.abspath(Save_H_URL)

    plt.subplot(121)
    plt.title("Recovered Random Pattern \n without Thresh")
    plt.imshow(H_Recover_Reshape_Threshed)
    plt.subplot(122)
    plt.title("Recovered Random Pattern \n with Thresh")
    plt.imshow(Random_Pattern_Correspondant_Reshape)
    plt.savefig(Save_H_URL_Absolute_Path)


H_Recover_Average_Overall_Sum = 0
for a in range(len(H_Recover_Average_List)):
    H_Recover_Average_Overall_Sum += float(H_Recover_Average_List[a])
H_Recover_Average_Overall = float(H_Recover_Average_Overall_Sum)/float(len(H_Recover_Average_List))

H_Recover_Median_Overall_Sum = 0
for b in range(len(H_Recover_Median_Full_List)):
    H_Recover_Median_Overall_Sum += float(H_Recover_Median_Full_List[b])
H_Recover_Median_Overall_Mean = float(H_Recover_Median_Overall_Sum)/float(len(H_Recover_Median_Full_List))

H_Recover_Median_Full_List_Sorted = sorted(H_Recover_Median_Full_List)
H_Recover_Median_Overall_Median = (float(H_Recover_Median_Full_List_Sorted[50]) + float(H_Recover_Median_Full_List_Sorted[51]))/float(2)

Save_Data_URL = "C:/Users/Dylan Luo/Documents/MMF Data (Updated with Recovered Images) - Dylan Luo/MMF data/Recovered Random Patterns/Random Patterns Data.txt"
Save_Data_URL_Absolute_Path = os.path.abspath(Save_Data_URL)
Data_Write = open(Save_Data_URL_Absolute_Path, "a")
Data_Write.write(str(H_Recover_Average_List))
Data_Write.write("\n \n" + str(H_Recover_Average_Overall))
Data_Write.write("\n \n" + str(H_Recover_Median_Full_List_Sorted))
Data_Write.write("\n \n" + str(H_Recover_Median_Overall_Mean))
Data_Write.write("\n \n" + str(H_Recover_Median_Overall_Median))
Data_Write.close()






