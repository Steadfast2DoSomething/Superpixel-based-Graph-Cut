# Superpixel-based-Graph-Cut

## 关于项目代码
项目暂时选用python作为主要开发代码，版本为2.7，后期会改用C++加速  
主要库包括不限于numpy，cv2(opencv for python)，skimage，pydicom，sklearn  
cv2 参考pdf 见参考文件文件夹  
skimage 文档：  http://scikit-image.org/docs/dev/  

## 主要思路：  
1. 预处理：</br>
1.1 定位到鼻咽癌相关层数和区域。（Q1，不同病人的z物理坐标不一致，需要定位特征点来定位层数;）</br>
1.2 预处理， 增强，去噪，平滑。（次要）</br>

2. Superpixel generation</br>
方法 SLIC （灰度距离+位置距离 聚类， 参数 超像素个数n，紧凑程度m);</br>
方法 Normalized Cuts  紧凑程度不可控;</br>
方法 Quick shift   超像素个数不可控;</br>
参考资料：http://ivrlwww.epfl.ch/supplementary_material/RK_SLICSuperpixels/index.html </br>
目前尝试通过实验找出最合适的超像素分割法。</br>

3. 特征提取和特征选择</br>
计划使用机器学习的方法来找到最佳的参数组合，例如：随机森林、SVM、深度学习等等；</br>
在上一步超像素的基础上进行特征提取，找出最能够反应超像素块特征的参数，分下一步分类打好基础。</br>

4. 分类器</br>
KNN SVM( with Gaussian kernal) RandomForest Boostmethod</br>
目的找到：</br>
计算p(1|s)和p(0|s)准确率最高的分类器和特征组合 （s是超像素快）</br>
异常检测：http://blog.csdn.net/l281865263/article/details/46654353</br>


5. 加入相邻像素影响</br>
Graph Cut</br>
如果graph cut 行不通，尝试MRF或者CRF的学习路线</br>  

6. 得到最终结果</br>

PS. 这里有一些机器学习、机器视觉的参考网站合集，分享给大家：  https://zhuanlan.zhihu.com/p/20787086</br>


## 关于code
使用前注意路径设置！
dicom data放在某盘根目录，自定
npy文件放在上一级目录 npy文件夹里 "..\NPY\"

readfile.py读入dcm信息，整合，
适用于处理较少数量的dcm文件，
处理数量较多的dcm文件时，建议先选择好范围，保存矩阵图片信息到npy文件，之后批量读入(类似于superpixel.py)

MRstrut.py

class myPatient
	类成员包括路径信息和MRsequence dict
	MRsequence dict中有四个成员，分别对应着四个不同的MR图
	函数 Mormalize 按照物理坐标对准四张图位置

class MRsequence
	最后的图片信息在cutImg中，医生勾画的结果保存在cutLabel中
	

readfile.py
	Thresholdsave   选定阈值保存
	SaveNpy 选定矩阵范围保存npy
	ShowLabelandContour  显示label和画在mr上的gtv
	SliceCluster  otsu二值化后，采用异或测量相邻两层距离
	
