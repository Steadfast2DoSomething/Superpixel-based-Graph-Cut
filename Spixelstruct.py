# --- update 2017/10/11 --- #

import numpy as np
import copy
import cv2


class Spixel:
    def __init__(self, imgData, superpixelLabel):
        self.img = imgData[:4,:,:]
        self.superpixelLabel = superpixelLabel
        self.GTVlabel = imgData[4,:,:]
        # print self.img.shape,
        # print self.GTVlabel.shape
        self.listnum = np.max(self.superpixelLabel) + 1
        self.sp_list = []

        # initial part
        self.GTVlabel[np.where(self.GTVlabel>0.9)] = 1
        self.GTVlabel[np.where(self.GTVlabel<=0.9)] = 0
        self.getOneSpixelList()

    def getOneSpixelList(self):
        for i in xrange(self.listnum):
            oneSpixel = OneSpixel()
            oneSpixel.Update_basicinfo(i, self.img, self.superpixelLabel, self.GTVlabel)
            self.sp_list.append(oneSpixel)
        print 'The number of sp of sp_list is:', len(self.sp_list)

class OneSpixel:
    def __init__(self):
        self.labelindex = -1
        self.mean = [-1,-1,-1,-1, 0]  #the fifth is average value
        self.variance = [-1,-1,-1,-1, 0]  #the fifth is average value


        self.centerpoint = [-1,-1]
        self.label = -1  # three option -1,0,1
        self.pixelnumber = -1
        self.limitsRange = [-1,-1,-1,-1] #Xmax, Xmin, Ymax, Ymin


    def Update_basicinfo(self, index, imgdata, labelmatrix, GTVlabel):
        self.labelindex = index
        targetindex = np.where(labelmatrix == index)
        #print index
        #--------------------------------------------------------------#
        # mean value and var value
        for i in xrange(4):
            temp = imgdata[i]
            self.mean[i] = np.mean(temp[targetindex])
            self.variance[i] = np.var(temp[targetindex])
        self.mean[4] = sum(self.mean)/4.
        self.variance[4] = sum(self.variance)/4.
        # number of pixel in one superpixel
        self.pixelnumber =  len(targetindex[0])

        #------------------------------------------------------------#
        '''label setting
        1 for tumor, 0 for normal tissue, -1 will not be used in training process
        partly because -1 labels are not typical tumor region
        '''
        # if and only if all of values in a superpixel are 1,
        # then the label of this superpixel is 1
        if 1 in GTVlabel[targetindex]:
            if 0 in GTVlabel[targetindex]:
                self.label = -1
            else:
                self.label = 1
        else:
            self.label = 0

        # ------------------------------------------------------------#
        # center_point
        self.centerpoint[0] = int(np.mean(targetindex[0]))
        self.centerpoint[1] = int(np.mean(targetindex[1]))
        #print self.centerpoint

        #------------------------------------------------------------#
        # contour_range
        self.limitsRange[0] = max(targetindex[0])
        self.limitsRange[1] = min(targetindex[0])
        self.limitsRange[2] = max(targetindex[1])
        self.limitsRange[3] = min(targetindex[1])

    def Feature_HoG(self):
        pass

    
    
