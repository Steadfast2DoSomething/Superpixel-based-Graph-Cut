# --- update 2017/10/12 --- #

from MRstruct import *
from Spixelstruct import *
from Graph import *
import cv2
import gc
# from sklearn.ensemble import RandomForestRegressor
import numpy as np
from skimage import data, segmentation, color
from skimage.future import graph
from sklearn import tree


def Superpixel_show(labels1, img, testImg):
    for x in xrange(1, labels1.shape[0]-1):
        for y in xrange(1, labels1.shape[1]-1):
            if labels1[x][y] != labels1[x][y+1] or labels1[x][y]!=labels1[x+1][y]:
                img[x, y, :] = 0
                if testImg[4,x,y] == 0:
                    testImg[4,x,y] = 1
                else:
                    testImg[4, x, y] = 0

    '''
    out1 = color.label2rgb(labels1, img[:, :, 0], kind="avg")
    out2 = color.label2rgb(labels1, img[:, :, 1], kind="avg")
    out3 = color.label2rgb(labels1, img[:, :, 2], kind="avg")
    out4 = color.label2rgb(labels1, img[:, :, 3], kind="avg")
    '''

    cv2.imshow("out1",pre_process(cv2.resize(img[:,:,0],(400,400))))
    cv2.imshow("out2",pre_process(cv2.resize(img[:,:,1],(400,400))))
    cv2.imshow("out3",pre_process(cv2.resize(img[:,:,2],(400,400))))
    cv2.imshow("out4",pre_process(cv2.resize(img[:,:,3],(400,400))))
    cv2.imshow("out5",pre_process(cv2.resize(testImg[4,:,:],(400,400))))


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def DataPro_Predict(allSp):
    DataPro = []
    return DataPro

def PairPro_Predict():
    pass

def GetconnecTwoSlices(slice1, slice2):
    row = len(slice1.sp_list)
    col = len(slice2.sp_list)
    connectMat = np.zeros((row,col))
    connectLabel = np.zeros((row,col))
    xlen,ylen= slice1.superpixelLabel.shape
    for x in xrange(xlen):
        for y in xrange(ylen):
            i = slice1.superpixelLabel[x][y]
            j = slice2.superpixelLabel[x][y]
            connectMat[i][j] = connectMat[i][j]+1
            if connectLabel[i][j]==0:
                a = slice1.sp_list[i].label
                b = slice2.sp_list[j].label
                if a == 1 and  b == 1:
                    connectLabel[i][j] = 1
                if a != 1 and b == 1:
                    connectLabel[i][j] = 2
                if a == 1 and b != 1:
                    connectLabel[i][j] = 2
                if a != 1 and b != 1:
                    connectLabel[i][j] = 3

    return connectMat, connectLabel

if __name__ == '__main__':
    allSp = []

    for i in xrange(1):
        print i
        temp = np.load("..\\NPY\\"+str(i)+'.npy')
        print temp.shape
        #--------normalize-----#
        maxvalue = np.max(temp[:4,:,:,:])
        for j in xrange(4):
            temp[j] = temp[j]/maxvalue*255.

        #-------------------------#
        tempSlice=None
        for z in xrange(temp.shape[1]):
            testImg = temp[:, z, 30:170, 30:170]  #further shrink  #second number for selecting one slice
            img = np.zeros((testImg.shape[1],testImg.shape[2],4))
            for j in xrange(4):
                img[:,:,j] = testImg[j]

            labels1 = segmentation.slic(img, compactness=70, n_segments=300)
            # Superpixel_show(labels1,img,testImg)
            oneSlice = Spixel(testImg,labels1)
            if tempSlice !=None:
                mat,label = GetconnecTwoSlices(tempSlice, oneSlice)
                oneSlice.connectMat = mat
                oneSlice.connectLabel = label
            tempSlice = oneSlice
            allSp.append(oneSlice.sp_list)

            
