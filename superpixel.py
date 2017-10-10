from MRstruct import *
from Spixelstruct import *
import cv2
import gc
# from sklearn.ensemble import RandomForestRegressor
import numpy as np
from skimage import data, segmentation, color
from skimage.future import graph
from sklearn import tree

def Superpixel_show(labels1, img,testImg):
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

if __name__ == '__main__':
    for i in xrange(1):
        print i
        temp = np.load(str(i)+'.npy')
        print temp.shape

        testImg = temp[:, 3, 30:170, 30:170]  #further shrink  #second number for selecting one slice
        print testImg.shape
        for j in xrange(4):
            pass
            # testImg[j] = cv2.GaussianBlur(testImg[j], (5,5), 0)

    img = np.zeros((testImg.shape[1],testImg.shape[2],4))
    for j in xrange(4):
        img[:,:,j] = testImg[j]

    labels1 = segmentation.slic(img, compactness=40, n_segments=500)
    #Superpixel_show(labels1,img,testImg)
    oneSlice = Spixel(testImg,labels1)


