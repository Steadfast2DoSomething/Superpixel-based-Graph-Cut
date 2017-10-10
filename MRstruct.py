import dicom
import numpy as np
import glob
import copy
import cv2


def Isintensify(contour):
    length = contour.shape
    length = length[0]
    for i in xrange(length-1):
        if abs(contour[i, 0] - contour[i+1, 0]) > 1 or abs(contour[i, 1] - contour[i+1,1]) > 1:
            print contour[i, 0], contour[i, 1], contour[i+1, 0], contour[i+1, 1],
            print "error", i, length
            return 0
    if contour[0,0] - contour[length-1,0] > 1 or contour[0, 1] - contour[length-1,1] >1:
        print "error", length, length
        return 0
    return 1

def IntensifyContour(contour):
    length = contour.shape
    length = length[0]
    zpos = contour[0,2]
    newList = []
    newList.append([contour[0, 0], contour[0, 1], zpos])
    for i in xrange(length-1):
        cx = contour[i, 0]; cy = contour[i, 1]
        nextx = contour[i+1, 0]; nexty = contour[i+1, 1];
        if cx == nextx and cy == nexty:
            continue
        elif abs(cx-nextx)>1.1 or abs(cy-nexty)>1.1:
            dis1 = abs(cx-nextx); dis2 = abs(cy-nexty)
            dis1 = max(dis1, dis2)
            vec1 = np.linspace(cx, nextx, dis1+1).astype(int)
            vec2 = np.linspace(cy, nexty, dis1+1).astype(int)
            for j in xrange(len(vec1)):
                newList.append([vec1[j], vec2[j], zpos])
        else:
            newList.append([cx, cy, zpos])
    cx = contour[length-1, 0]; cy = contour[length-1, 1]
    nextx = contour[0, 0]; nexty = contour[0, 1]

    if cx == nextx and cy == nexty:
        return newList
    elif abs(cx - nextx) > 1.1 or abs(cy - nexty) > 1.1:
        dis1 = abs(cx - nextx)
        dis2 = abs(cy - nexty)
        dis1 = max(dis1, dis2)
        vec1 = np.linspace(cx, nextx, dis1 + 1).astype(int)
        vec2 = np.linspace(cy, nexty, dis1 + 1).astype(int)
        for j in xrange(len(vec1)):
            newList.append([vec1[j], vec2[j], zpos])
    else:
        newList.append([cx, cy, zpos])
    return newList

def Imgfill(rect):
    w, h = rect.shape
    rect[0][0] = 2; rect[0][h-1] = 2
    rect[w - 1][h - 1] = 2; rect[w - 1][0] = 2
    xx = [0, 0, w - 1, w - 1];
    yy = [0, h - 1, h - 1, 0]

    while True:
        new_xx = []
        new_yy = []
        for i in xrange(len(xx)):
            x = xx[i]; y = yy[i]
            if (x-1) > -1:
                if rect[x-1][y] == 0:
                    rect[x-1][y] = 2
                    new_xx.append(x-1)
                    new_yy.append(y)

            if (x+1) < w:
                if rect[x+1][y] == 0:
                    rect[x+1][y] = 2
                    new_xx.append(x+1)
                    new_yy.append(y)

            if (y-1) > -1:
                if rect[x][y-1] == 0:
                    rect[x][y-1] = 2
                    new_xx.append(x)
                    new_yy.append(y-1)

            if (y+1) < h:
                if rect[x][y+1] == 0:
                    rect[x][y+1] = 2
                    new_xx.append(x)
                    new_yy.append(y+1)

        if len(new_yy) == 0:
            break
        xx = new_xx; yy = new_yy

    num1 = len(np.where(rect == 2)[0])
    num2 = len(np.where(rect == 0)[0])
    rect[np.where(rect == 0)] = 1
    rect[np.where(rect == 2)] = 0
    return rect

def pre_process(image):

    """
    convert to range 0-255 for display
    :rtype: object
    """
    max_v = np.max(image)
    min_v = np.min(image)
    if max_v == min_v:
        return image
    image = (image - min_v) * 255. / (max_v - min_v)
    image = np.uint8(image)
    return image


class myPatient:
    def __init__(self,filepath):
        #class memember

        self.basefilepath = ""
        self.MRfilelist = [] # default empty list
        self.MRsequence= {} # default empty dict

        # ---------------------------------------------------------------- #
        # initialize part
        self.basefilepath = filepath

    def Dataupdate(self):
        self.GetMRfilelist()
        self.SetMRsequenceData()
        self.Normalize()

    def GetMRfilelist(self):
        self.MRfilelist = glob.glob(self.basefilepath+"*")
        if len(self.MRfilelist) != 4:
            print "wrong filepath!!"
            exit(1)
        print 'Show MRfilelist as follows: '
        print self.MRfilelist
        print "--------------------------"

    def SetMRsequenceData(self):
        for i in xrange(4):
            self.MRsequence[i] = MRsequence(self.MRfilelist[i])

    def Normalize(self):
        left_up = []
        right_down = []
        for i in xrange(4):
            obj = self.MRsequence[i]
            left_up.append(obj.up_left_pos)
            tempx = obj.pixelspace * obj.size[0] + obj.up_left_pos[0]
            tempy = obj.pixelspace * obj.size[1] + obj.up_left_pos[1]
            right_down.append([tempx, tempy])
        left_up = np.array(left_up)
        right_down = np.array(right_down)
        newlu = [max(left_up[:, 0]), max(left_up[:, 1])]
        newrd = [min(right_down[:, 0]), min(right_down[:, 1])]

        for i in xrange(4):
            obj = self.MRsequence[i]
            pixspace = obj.pixelspace
            pos = obj.up_left_pos
            rangex1 = int((newlu[0] - pos[0]) / pixspace)
            rangey1 = int((newlu[1] - pos[1]) / pixspace)
            rangex2 = int((newrd[0] - pos[0]) / pixspace)
            rangey2 = int((newrd[1] - pos[1]) / pixspace)
            scale1 = rangex2-rangex1
            scale2 = rangey2-rangey1
            print 'The value of rangex1 is:', rangex1, ', the value of rangey1 is:', rangey1
            print 'The value of rangex2 is:', rangex2, ', the value of rangey2 is:', rangey2

            cutImg = obj.ImageData[:, rangex1:rangex2+1, rangey1:rangey2+1]
            newImg = []
            newLabel = []
            for j in xrange(100):
                temp = cv2.resize(cutImg[j],(600,int(600*scale1/float(scale2))))
                newImg.append(temp)
                temp2 = cv2.resize(obj.LabelData[j],(600,int(600*scale1/float(scale2))))
                newLabel.append(temp2)

            self.MRsequence[i].cutImg = np.array(newImg)
            self.MRsequence[i].cutLabel = np.array(newLabel)


class MRsequence():
    def __init__(self, filedirectory):
        # class memember discare
        self.cutImg = None
        self.cutLabel = None
        self.filepath = " "
        self.dcmlist = [] # default empty list
        self.strcstr = " "
        self.dcmnumber = -1
        self.up_left_pos = [0, 0]  # default 0,0
        self.Zrange = [0, 0]

        self.GTVData = np.array((1, 1))
        self.GTVzpos_range = [0, 0]
        self.GTVslicenum = -1
        self.GTVpixData = []
        self.GTVlimit = [0, 0, 0, 0]
        # initial part

        self.filepath = filedirectory
        self.dcmlist = glob.glob(self.filepath+"\\*image*.dcm")
        self.strcstr = glob.glob(self.filepath+"\\*str*.dcm")
        self.strcstr = self.strcstr[0]
        self.dcmnumber = len(self.dcmlist)
        self.Zposlist = []  # index to zpos
        # self.Zindexdict = {} # zpos to index
        self.pixelspace = 0;
        self.size = [0, 0]
        self.ImageData = None

        array = []
        for name in self.dcmlist:
            info = dicom.read_file(name, force = True)
            tempz = info.ImagePositionPatient[2]
            self.Zposlist.append(tempz)
            array.append(info.pixel_array)

        self.size = array[0].shape
        self.ImageData = np.zeros((100, self.size[0], self.size[1]))  # from -99 to 198 with space 3
        for i in xrange(len(self.Zposlist)):
            temp = int(round((self.Zposlist[i]+99)/3.))
            if temp < 0 or temp >= 100:
                continue
            self.ImageData[temp, :, :] = array[i]

        self.LabelData = np.zeros((100, self.size[0], self.size[1]))
        # self.Zposlist.sort() # ascend order
        # self.Zrange = [self.Zposlist[0], self.Zposlist[-1]]
        # print self.Zindexdict
        self.up_left_pos = info.ImagePositionPatient[0:2]
        self.pixelspace = float(info.PixelSpacing[0])
        # self.up_left_pos = float(self.up_left_pos)
        # cv2.imshow(self.strcstr,pre_process(self.ImageData[40]))
        print 'The value of up_left_pos is:', self.up_left_pos
        print 'The value of pixelspace is:', self.pixelspace
        print 'The value of size[0] is:', self.size[0]
        print 'The value of the p * size[0] + u[0] is:', self.pixelspace*int(self.size[0])+self.up_left_pos[0]
        print 'The value of the p * size[1] + u[1] is:', self.pixelspace*int(self.size[1])+self.up_left_pos[1]
        print "--------------------------"

        #print self.up_left_pos


        # --------------GTV data part-------------------------------- #

        self.SetGTVData()
        self.GetLabelData()

    def SetGTVData(self):
        info = dicom.read_file(self.strcstr, force=True)
        self.GTVslicenum = len(info.ROIContourSequence[0].ContourSequence)
        # print contournum
        contourData = []
        tempZlist = []
        for i in xrange(self.GTVslicenum):
            contourData.append(np.reshape(info.ROIContourSequence[0].ContourSequence[i].ContourData, (-1, 3)))
            tempZlist.append(contourData[i][0,2])
        self.GTVzpos_range = [min(tempZlist), max(tempZlist)]
        self.GTVData = np.array(contourData)
        print 'The value of strct is: ',max(self.GTVData[self.GTVslicenum-5][:,0]), max(self.GTVData[self.GTVslicenum-5][:,1]) # check contour coordinate in different MRI

    def GetLabelData(self):
        self.GTVpixData = copy.deepcopy(self.GTVData)
        # print self.GTVpixData[1]
        for i in xrange(self.GTVslicenum):
            length = self.GTVpixData[i].shape
            length = length[0]
            # physical position to pixel index
            for j in xrange(length):
                self.GTVpixData[i][j][0] = int((self.GTVpixData[i][j, 0] - self.up_left_pos[0]) / self.pixelspace)
                self.GTVpixData[i][j][1] = int((self.GTVpixData[i][j, 1] - self.up_left_pos[1]) / self.pixelspace)
                self.GTVpixData[i][j][2] = int(round((self.GTVpixData[i][j, 2]+99)/3.))
            # self.GTVpixData[i] = np.unique(self.GTVpixData[i],axis=0)
            self.GTVpixData[i].astype(int)
            # rule out pixel at far away distance
            self.GTVpixData[i] = np.array(IntensifyContour(self.GTVpixData[i]))
            # print Isintensify(self.GTVpixData[i])
            # self.GTVpixData[i] = np.unique(self.GTVpixData[i].view(self.GTVpixData[i].dtype.descr * self.GTVpixData[i].shape[1]))

        #print self.GTVpixData[1]
        #print np.array(self.GTVpixData[1])
        # print self.GTVpixData[i]
        for i in xrange(self.GTVslicenum):
            if self.GTVpixData[i].shape[0]==0:
                continue
            zindex = self.GTVpixData[i][0][2]
            # w,h = self.LabelData[i].shape
            length = self.GTVpixData[i].shape
            length = length[0]
            ymax = 0; ymin = 1000; xmax = 0; xmin = 1000
            for j in xrange(length):
                if ymax < self.GTVpixData[i][j, 1]:
                    ymax = self.GTVpixData[i][j, 1]
                elif ymin > self.GTVpixData[i][j, 1]:
                    ymin = self.GTVpixData[i][j, 1]

                if xmax < self.GTVpixData[i][j, 0]:
                    xmax = self.GTVpixData[i][j, 0]

                elif xmin > self.GTVpixData[i][j, 0]:
                    xmin = self.GTVpixData[i][j, 0]

            self.GTVlimit = [xmin, xmax, ymin, ymax]

            zindex = int(self.GTVpixData[i][0, 2])
            for j in xrange(length):
                self.LabelData[zindex][int(self.GTVpixData[i][j][1])][int(self.GTVpixData[i][j][0])] = 1

            xmin, ymin = ymin, xmin
            xmax, ymax = ymax, xmax
            rect = self.LabelData[zindex][int(xmin)-2:int(xmax) + 3, int(ymin)-2:int(ymax) + 3]
            w,h = rect.shape
            if w>2 and h>2:
                rect = Imgfill(rect)
            self.LabelData[zindex][int(xmin) - 2:int(xmax) + 3, int(ymin) - 2:int(ymax) + 3] = rect
