# Introduction

They are codes used in my Bachelor Thesis, which has the same title with this repository. The result is seemingly not good enough, so I still have some work to do.



Please feel free to read and try this codes. It may be a good **reference** for you, but not a good and stable implement. On the other word, it is hard to use this codes **directly** in your **scenario**. :(

# Method

## Multi-Frame-Based

In this method, I want to solve the following problem:

![Equ-MultiframeAll](https://raw.githubusercontent.com/WJGan1995/Video-Super-Resolution/master/result/Equ-MultiframeAll.png)

which can be splited into two sub-problem like this:

(1) **MAP & Total Variation Minimum**:  

![Equ-MultiframeAll](https://raw.githubusercontent.com/WJGan1995/Video-Super-Resolution/master/result/Equ-MAPandTV.png)

(2) **L2-Sparse**: 

![Equ-MultiframeAll](https://raw.githubusercontent.com/WJGan1995/Video-Super-Resolution/master/result/Equ-L2Sparse.png)



I solve the first sub-problem with steepest descent method (Actually **Adam**) with the help of **tensorflow**. And I solve the second sub-problem with GPSR([Gradient Projection for Sparse Representation](http://www.lx.it.pt/~mtf/GPSR/)) in Matlab. I implement above method in each frame of video.



After that, I use **Total Variation Minimum in time domain** to deblur video, which means:

![Equ-MultiframeAll](https://raw.githubusercontent.com/WJGan1995/Video-Super-Resolution/master/result/Equ-TVinTime.png)

$h$ means video obtained in above step. How to solve it? : [deconvtv - fast algorithm for total variation deconvolution](https://ww2.mathworks.cn/matlabcentral/fileexchange/43600-deconvtv-fast-algorithm-for-total-variation-deconvolution?s_tid=srchtitle)

## DeepLearning-Based

Just a simple implement of [SRCNN](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html). There maybe a better implement: [SRCNN-Tensorflow](https://github.com/tegg89/SRCNN-Tensorflow]). Though my code is simple, but its code may be more understandable.



My train data is from here: [NTIRE2017](http://www.vision.ee.ethz.ch/ntire17/), which is too big. You must download it by yourself.

# Files Organization

I fulfill my method with the help of `Python` and `Matlab`. Actually, I prefer `Matlab` more. But when it comes to Deep-Learning, Python is better. So… :)

- **Deconvtvl2.m**: A implement of Total Variation Minimum in time domain, got from [deconvtv - fast algorithm for total variation deconvolution](https://ww2.mathworks.cn/matlabcentral/fileexchange/43600-deconvtv-fast-algorithm-for-total-variation-deconvolution?s_tid=srchtitle)
- **GPSRBasic.m**: A implement of L2-Sparse, got from [Gradient Projection for Sparse Representation](http://www.lx.it.pt/~mtf/GPSR/)
- **Multiframe.m**: When I finish MAP&Total Variation Mnimum in Pyhon, I continue my multiframe-based method in this file, with the help of above two files.
- **inputvideo.py**: How to input video can convert them into many groups with Python and OpenCV
- **maptv.py**: A implement of MAP&Total Variation Mnimum in python.
- **motion.py**: How to calculate motion matrix in multiframe-based method
- **srcnn.py**: A implement of SRCNN.



Also, my software environment:

- **Python**: 3.6.4
- **OpenCV for python**: 3.4
- **Tensorflow**: 1.2-gpu
- **Matlab**: R2018a 

# FInal Result

![There may be some troubles. :(](https://raw.githubusercontent.com/WJGan1995/Video-Super-Resolution/master/result/Result-showed-in-GitHub.png "Figure: Image of final result")

# How to use

I share my code with the aim of giving other people some refered material. So, if indeed you have insterested in my code, **to read and understand it**. And then you will know how to use it. :) 

Just notice one thing: I do not write a file like `main.py`. The realization of the *Multi-Frame-Based* method is in the bottom of `maptv.py` and in `Multiframe.m`, while the *DeepLearning-Based* method is in the bottom of `srcnn.py`