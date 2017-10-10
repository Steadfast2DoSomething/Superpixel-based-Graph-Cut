import dicom
import numpy as np
import glob
import copy
import cv2
import os
from MRstruct import *
import gc
import matplotlib.pyplot as plt

basefilepath1 = "F:\\GTVsegment_Data\\first_data0717(172)\\"
basefilepath2 = "F:\\GTVsegment_Data\\2017-08-02(255)\\"
filelist1 = glob.glob(basefilepath1 + "*")
filelist2 = glob.glob(basefilepath2 + "*")

centerSize = [170, 370, 200, 400]


def ThresholdSave(th1, th2):
    i = 0
    for name in filelist1[:10]:
        i = i +1
        try:
            os.mkdir("th"+str(i))
        except:
            pass
        mrPatient = myPatient(name+"\\")
        mrPatient.Dataupdate()
        for j in xrange(1): #loop in four MRI
            oneMRs = mrPatient.MRsequence[j]
            Img = oneMRs.cutImg
            print "------------------"
            slice = Img.shape
            slice = slice[0]
            maxvalue = np.max(oneMRs.cutImg)
            #minvalue = np.min(oneMRs.cutImg)
            #Img[np.where(Img==0)] = 1000
            Img[np.where(Img<=th1)] = 0
            #Img[np.where(Img==1000)] = 0
            Img[np.where(Img>th1)] = 1
            for z in xrange(10,slice-10):
                if 1 in Img[z]:
                    kernal = np.ones((3,3))
                    #Img[z] = cv2.dilate(Img[z], kernal, iterations=2)
                    #Img[z] = cv2.erode( Img[z], kernal, iterations=2)
                    tempfilename = "th"+str(i)+"\\t1-"+str(z)+".tif"
                    # cv2.imshow(str(z),Img[z])
                    # save change operation
                    temp = Img[z]
                    temp[np.where(temp==1) ]=255
                    cv2.imwrite(tempfilename, temp)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def SaveNpy():
    i = 0
    for name in filelist1[1:3]:
        mrPatient = myPatient(name+"\\")
        mrPatient.Dataupdate()
        oneData = []
        for j in xrange(1):
            oneMRs = mrPatient.MRsequence[j]
            print "-------------------------"
            # for z in xrange(38,46):
            cv2.imshow(name+str(40),\
                       pre_process(oneMRs.cutImg[40,centerSize[0]:centerSize[1],centerSize[2]:centerSize[3]]))
            oneData.append(oneMRs.cutImg[38:46,centerSize[0]:centerSize[1],centerSize[2]:centerSize[3]])
        oneData.append(oneMRs.cutLabel[38:46,centerSize[0]:centerSize[1],centerSize[2]:centerSize[3]])
        oneData = np.array(oneData)
        # print oneData.shape
        oneData.astype(int)
        # np.save(str(i)+".npy",oneData)
        del oneData
        del mrPatient
        gc.collect()
        i = i + 1
        print i
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ShowLabelandContour():
    i = 0
    for name in filelist1[0:9]:
        i = i + 1
        print i,name
        mrPatient = myPatient(name + "\\")
        mrPatient.Dataupdate()
        oneData = []
        for j in xrange(4):# loop in 4 MRI
            oneMRs = mrPatient.MRsequence[j]
            print "-------------------------"
            gtvsliceNum = oneMRs.GTVslicenum
            print gtvsliceNum,"gtvslicenum"
            for z in xrange(gtvsliceNum):
                length = oneMRs.GTVpixData[z].shape
                length = length[0]
                tmpz =  int(oneMRs.GTVpixData[z][0,2])
                print tmpz
                #tmpz = int(oneMRs.GTVpixData[z][0,2]+99/3.)
                tmpImg = oneMRs.ImageData[tmpz]
                for k in xrange(length-1):
                    cv2.line(tmpImg, (int(oneMRs.GTVpixData[z][k,0]), int(oneMRs.GTVpixData[z][k,1])), \
                             (int(oneMRs.GTVpixData[z][k+1,0]), int(oneMRs.GTVpixData[z][k+1,1])), (0,0,255))

                #cv2.imshow(str(tmpz)+" "+name, pre_process(oneMRs.ImageData[tmpz]))
                #print str(tmpz)+" "+name+".png"
                print cv2.imwrite("p"+str(i)+"T"+str(j)+"contour"+str(tmpz)+".png", pre_process(oneMRs.ImageData[tmpz]))
                #cv2.imshow(str(tmpz)+"//"+name, oneMRs.LabelData[tmpz])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def SliceCluster():
    i = 0
    for name in filelist1[:1]:
        i = i + 1
        mrPatient = myPatient(name + "\\")
        mrPatient.Dataupdate()
        for j in xrange(1):  # loop in four MRI
            oneMRs = mrPatient.MRsequence[j+3]
            Img = oneMRs.cutImg
            print "------------------"
            slice = Img.shape
            print "The value of slice is: ", 
            print slice
            slice = slice[0]
            maxvalue = np.max(oneMRs.cutImg)
            # minvalue = np.min(oneMRs.cutImg)
            # Img[np.where(Img==0)] = 1000
            Canny = copy.copy(Img)
            Otsu = np.zeros((slice-20,200,200))
            for z in xrange(10, slice-20):
                #Canny[z] = cv2.Canny(np.uint8(Img[z]),100,200)
                #cv2.imshow(str(z),Canny[z])
                ret,Otsu[z] = cv2.threshold(np.uint8(cv2.resize(Img[z],(200,200))),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                kernal = np.ones((1, 1))
                Otsu[z] = cv2.dilate( Otsu[z], kernal, iterations=1)
                Otsu[z] = cv2.erode( Otsu[z], kernal, iterations=1)
                cv2.imshow(str(z),Otsu[z])
            histdata = []
            for z in xrange(10, slice - 21):
                histdata.append(np.sum(np.logical_xor(Otsu[z],Otsu[z+1])))

            plt.plot(histdata)
            plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    print "Here we goooooo!!!!!!"
    # ShowLabelandContour()
    #ThresholdSave(50,100)
    SliceCluster()

