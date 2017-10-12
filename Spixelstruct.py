# --- update 2017/10/12 --- #

import numpy as np
import copy
import cv2
import math


class Spixel:
    def __init__(self, imgData, superpixelLabel):
        self.img = imgData[:4,:,:]
        self.superpixelLabel = superpixelLabel
        self.GTVlabel = imgData[4,:,:]
        # print self.img.shape,
        # print self.GTVlabel.shape
        self.listnum = np.max(self.superpixelLabel) + 1
        self.sp_list = []
        self.connectMat = None   # compare with z-1
        self.connectLabel = None # compare with z-1
        # initial part
        self.GTVlabel[np.where(self.GTVlabel>0.9)] = 1
        self.GTVlabel[np.where(self.GTVlabel<=0.9)] = 0
        self.getOneSpixelList()
        self.edges, self.edgeslabel  = self.GetPairwiseEdgeAndLabel()
        print self.edgeslabel

    def getOneSpixelList(self):
        for i in xrange(self.listnum):
            oneSpixel = OneSpixel()
            oneSpixel.Update_basicinfo(i, self.img, self.superpixelLabel, self.GTVlabel)
            oneSpixel.Spixel_GrayHist(self.img, self.superpixelLabel)
            oneSpixel.Spixel_entro(self.img,self.superpixelLabel)
            # print oneSpixel.entro
            self.sp_list.append(oneSpixel)

    # @output
    # edges: pairwise matrix
    def GetPairwiseEdgeAndLabel(self):
        [row, col] = self.superpixelLabel.shape
        edges = np.zeros((self.listnum, self.listnum))
        edgesLabel = np.zeros((self.listnum, self.listnum))

        for i in xrange(0, row - 1):
            for j in xrange(0, col - 1):
                if (self.superpixelLabel[i][j] != self.superpixelLabel[i][j + 1]):
                    edges[self.superpixelLabel[i][j]][self.superpixelLabel[i][j + 1]] = 1
                    edges[self.superpixelLabel[i][j + 1]][self.superpixelLabel[i][j]] = 1
                if (self.superpixelLabel[i][j] != self.superpixelLabel[i + 1][j]):
                    edges[self.superpixelLabel[i][j]][self.superpixelLabel[i + 1][j]] = 1
                    edges[self.superpixelLabel[i + 1][j]][self.superpixelLabel[i][j]] = 1



        for i in xrange(self.listnum):
            for j in xrange(i,self.listnum):
                if edges[i][j]==1:
                    a = self.sp_list[i].label
                    b = self.sp_list[j].label
                    if a==1 and b== 1:
                        edgesLabel[i][j] = 1
                    if a != 1 and b == 1:
                        edgesLabel[i][j] = 2
                    if a == 1 and b != 1:
                        edgesLabel[i][j] = 2
                    if a != 1 and b != 1:
                        edgesLabel[i][j] = 3
        return edges, edgesLabel

class OneSpixel:
    def __init__(self):
        self.labelindex = -1
        self.mean = [-1,-1,-1,-1, 0]  # the fifth is average value
        self.variance = [-1,-1,-1,-1, 0]  # the fifth is average value
        self.coe_var = [-1,-1,-1,-1]

        self.centerxy = [-1,-1]
        self.label = -1  # three options -1,0,1
        self.pixelnumber = -1
        self.limitsRange = [-1,-1,-1,-1] # Xmax, Xmin, Ymax, Ymin
        self.grayhist = []
        self.entro = []

    def Update_basicinfo(self, index, imgdata, labelmatrix, GTVlabel):
        self.labelindex = index
        targetindex = np.where(labelmatrix == index)
        print index
        #--------------------------------------------------------------#
        # mean value ,var value, coefficient of var
        for i in xrange(4):
            temp = imgdata[i]
            self.mean[i] = np.mean(temp[targetindex])
            self.variance[i] = np.var(temp[targetindex])
            self.coe_var[i] = math.sqrt(self.variance[i]) / self.mean[i]

        self.mean[4] = sum(self.mean)/4.
        self.variance[4] = sum(self.variance)/4.
        # number of pixel in one superpixel
        self.pixelnumber =  len(targetindex[0])
        # print self.mean
        # print self.variance
        # print self.coe_var

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
        self.centerxy[0] = int(np.mean(targetindex[0]))
        self.centerxy[1] = int(np.mean(targetindex[1]))
        print self.centerxy

        #------------------------------------------------------------#
        # contour_range
        self.limitsRange[0] = max(targetindex[0])
        self.limitsRange[1] = min(targetindex[0])
        self.limitsRange[2] = max(targetindex[1])
        self.limitsRange[3] = min(targetindex[1])

    def Feature_HoG(self):
        pass

    def Spixel_GrayHist(self, imgdata, labelmatrix, histlen=8):
        # default 8
        targetindex = np.where(labelmatrix == self.labelindex)
        temp = [0]*histlen
        for i in xrange(self.pixelnumber):
            for j in xrange(4):
                val = imgdata[j, targetindex[0][i], targetindex[1][i]]
                x = int(val*histlen/255.)
                if x==histlen:
                    x = histlen-1
                temp[x] = temp[x]+1

        self.grayhist = [v/4.0/self.pixelnumber for v in temp]


    def Spixel_entro(self, imgdata, labelmartix,f_len=16):
        targetindex = np.where(labelmartix==self.labelindex)
        entro = [0]*4
        for i in xrange(4):
            temp = [0]*f_len
            img = imgdata[i]
            for j in xrange(self.pixelnumber):
                val = img[targetindex[0][j],targetindex[1][j]]
                x = int(val*f_len/255.)
                if x == f_len:
                    x = f_len-1
                temp[x] = temp[x]+1
            #print temp
            temp = [1.0*v/self.pixelnumber for v in temp]
            for j in xrange(f_len):
                if temp[j]==0:
                    continue
                entro[i] = entro[i]-temp[j]*math.log(temp[j])

        self.entro = entro


    
