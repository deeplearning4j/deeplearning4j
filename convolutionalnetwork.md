---
title: A Beginner's Guide to Convolutional Networks
layout: default
redirect_from: convolutionalnets
---

# A Beginner's Guide to Convolutional Networks

Contents

* <a href="#intro">Convolutional Net Introduction</a>
* <a href="#tensors">Images Are 4-D Tensors?</a>
* <a href="#define">ConvNet Definition</a>
* <a href="#work">How Convolutional Nets Work</a>
* <a href="#max">Maxpooling/Downsampling</a>
* <a href="#code">DL4J Code Sample</a>
* <a href="#resource">Other Resources</a>

## <a name="intro">Introduction to Convolutional Networks</a>

Convolutional networks are deep artificial neural networks that can be used to classify images (name what they see), cluster them by similarity (photo search), and perform object recognition within scenes. They are algorithms that can identify faces, individuals, street signs, eggplants, platypuses and many other aspects of visual data. 

Convolutional networks perform optical character recognition (OCR) to digitize text and make natural-language processing possible on analog and hand-written documents, where the images are symbols to be transcribed. CNNs can also be applied to sound when it is represented visually as a spectrogram. More recently, convolutional networks have been applied directly to text analytics as well as graph data with [graph convolutional networks](./graphdata). 

The efficacy of convolutional nets (ConvNets or CNNs) in image recognition is one of the main reasons why the world has woken up to the efficacy of deep learning. They are powering major advances in machine vision, which has obvious applications for self-driving cars, robotics, drones, security, medical diagnoses, and treatments for the visually impaired. 

Skymind wraps NVIDIA's cuDNN and integrates with OpenCV. Our convolutional nets run on distributed GPUs using Spark, making them among the fastest in the world. You can learn how to build a [image recognition web app with VGG16 here](./build_vgg_webapp) and how to [deploy CNNs to Android here](./android).

<p align="center">
<a href="https://docs.skymind.ai/docs/welcome" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">GET STARTED WITH CONVOLUTIONAL NETWORKS</a>
</p>

## <a name="tensors">Images Are 4-D Tensors?</a>

Convolutional nets ingest and process images as tensors, and tensors are matrices of numbers with additional dimensions. 

They can be hard to visualize, so let’s approach them by analogy. A scalar is just a number, such as 7; a vector is a list of numbers (e.g., `[7,8,9]`); and a matrix is a rectangular grid of numbers occupying several rows and columns like a spreadsheet. Geometrically, if a scalar is a zero-dimensional point, then a vector is a one-dimensional line, a matrix is a two-dimensional plane, a stack of matrices is a three-dimensional cube, and when each element of those matrices has a stack of *feature maps* atttached to it, you enter the fourth dimension. For reference, here’s a 2 x 2 matrix:

    [ 1, 2 ] 
    [ 5, 8 ]

A tensor encompasses the dimensions beyond that 2-D plane. You can easily picture a three-dimensional tensor, with the array of numbers arranged in a cube. Here’s a 2 x 3 x 2 tensor presented flatly (picture the bottom element of each 2-element array extending along the z-axis to intuitively grasp why it’s called a 3-dimensional array):

![Alt text](./img/tensor.png) 

In code, the tensor above would appear like this: `[[[2,3],[3,5],[4,7]],[[3,4],[4,6],[5,8]]].` And here's a visual: 

![Alt text](./img/3d_matrix_cube.png) 

In other words, tensors are formed by arrays nested within arrays, and that nesting can go on infinitely, accounting for an arbitrary number of dimensions far greater than what we can visualize spatially. A 4-D tensor would simply replace each of these scalars with an array nested one level deeper. Convolutional networks deal in 4-D tensors like the one below (notice the nested array).

![Alt text](./img/3d_matrix.png) 

ND4J and Deeplearning4j use `NDArray` synonymously with tensor. A tensor’s dimensionality `(1,2,3…n)` is called its order; i.e. a fifth-order tensor would have five dimensions.

The width and height of an image are easily understood. The depth is necessary because of how colors are encoded. Red-Green-Blue (RGB) encoding, for example, produces an image three layers deep. Each layer is called a "channel", and through convolution it produces a stack of feature maps (explained below), which exist in the fourth dimension, just down the street from time itself. (Features are just details of images, like a line or curve, that convolutional networks create maps of.)

So instead of thinking of images as two-dimensional areas, in convolutional nets they are treated as four-dimensional volumes. These ideas will be explored more thoroughly below. 

## <a name="define">Definition</a>

From the Latin *convolvere*, "to convolve" means to roll together. For mathematical purposes, a convolution is the integral measuring how much two functions overlap as one passes over the other. Think of a convolution as a way of mixing two functions by multiplying them. 

<!-- <iframe src="http://mathworld.wolfram.com/images/gifs/convgaus.gif" width="100%" height="260px;" style="border:none;"></iframe>
-->
<iframe src="img/convgaus.gif" width="100%" height="260px;" style="border:none;"></iframe>
*Credit: [Mathworld](http://mathworld.wolfram.com/). "The green curve shows the convolution of the blue and red curves as a function of t, the position indicated by the vertical green line. The gray region indicates the product `g(tau)f(t-tau)` as a function of t, so its area as a function of t is precisely the convolution."*

Look at the tall, narrow bell curve standing in the middle of a graph. The integral is the area under that curve. Near it is a second bell curve that is shorter and wider, drifting slowly from the left side of the graph to the right. The product of those two functions' overlap at each point along the x-axis is their [convolution](http://mathworld.wolfram.com/Convolution.html). So in a sense, the two functions are being "rolled together." 

With image analysis, the static, underlying function (the equivalent of the immobile bell curve) is the input image being analyzed, and the second, mobile function is known as the filter, because it picks up a signal or feature in the image. The two functions relate through multiplication. To visualize convolutions as matrices rather than as bell curves, please see [Andrej Karpathy's excellent animation](https://cs231n.github.io/convolutional-networks/) under the heading "Convolution Demo."

The next thing to understand about convolutional nets is that they are passing *many* filters over a single image, each one picking up a different signal. At a fairly early layer, you could imagine them as passing a horizontal line filter, a vertical line filter, and a diagonal line filter to create a map of the edges in the image. 

Convolutional networks take those filters, slices of the image's feature space, and map them one by one; that is, they create a map of each place that feature occurs. By learning different portions of a feature space, convolutional nets allow for easily scalable and robust feature engineering.

(Note that convolutional nets analyze images differently than RBMs. While RBMs learn to reconstruct and identify the features of each image as a whole, convolutional nets learn images in pieces that we call feature maps.) 

So convolutional networks perform a sort of search. Picture a small magnifying glass sliding left to right across a larger image, and recommencing at the left once it reaches the end of one pass (like typewriters do). That moving window is capable recognizing only one thing, say, a short vertical line. Three dark pixels stacked atop one another. It moves that vertical-line-recognizing filter over the actual pixels of the image, looking for matches.

Each time a match is found, it is mapped onto a feature space particular to that visual element. In that space, the location of each vertical line match is recorded, a bit like birdwatchers leave pins in a map to mark where they last saw a great blue heron. A convolutional net runs many, many searches over a single image – horizontal lines, diagonal ones, as many as there are visual elements to be sought. 

Convolutional nets perform more operations on input than just convolutions themselves. 

After a convolutional layer, input is passed through a nonlinear transform such as *tanh* or *rectified linear* unit, which will squash input values into a range between -1 and 1. 

## <a name="work">How Convolutional Networks Work</a>

The first thing to know about convolutional networks is that they don't perceive images like humans do. Therefore, you are going to have to think in a different way about what an image means as it is fed to and processed by a convolutional network. 

Convolutional networks perceive images as volumes; i.e. three-dimensional objects, rather than flat canvases to be measured only by width and height. That's because digital color images have a red-blue-green (RGB) encoding, mixing those three colors to produce the color spectrum humans perceive. A convolutional network ingests such images as three separate strata of color stacked one on top of the other. 

So a convolutional network receives a normal color image as a rectangular box whose width and height  are measured by the number of pixels along those dimensions, and whose depth is three layers deep, one for each letter in RGB. Those depth layers are referred to as *channels*. 

As images move through a convolutional network, we will describe them in terms of input and output volumes, expressing them mathematically as matrices of multiple dimensions in this form: 30x30x3. From layer to layer, their dimensions change for reasons that will be explained below. 

You will need to pay close attention to the precise measures of each dimension of the image volume, because they are the foundation of the linear algebra operations used to process images. 

Now, for each pixel of an image, the intensity of R, G and B will be expressed by a number, and that number will be an element in one of the three, stacked two-dimensional matrices, which together form the image volume. 

Those numbers are the initial, raw, sensory features being fed into the convolutional network, and the ConvNets purpose is to find which of those numbers are significant signals that actually help it classify images more accurately. (Just like other feedforward networks we have discussed.)

Rather than focus on one pixel at a time, a convolutional net takes in square patches of pixels and passes them through a *filter*. That filter is also a square matrix smaller than the image itself, and equal in size to the patch. It is also called a *kernel*, which will ring a bell for those familiar with support-vector machines, and the job of the filter is to find patterns in the pixels. 

<iframe src="https://cs231n.github.io/assets/conv-demo/index.html" width="100%" height="700px;" style="border:none;"></iframe>
*Credit for this excellent animation goes to [Andrej Karpathy](https://cs231n.github.io/).*

Imagine two matrices. One is 30x30, and another is 3x3. That is, the filter covers one-hundredth of one image channel's surface area. 

We are going to take the dot product of the filter with this patch of the image channel. If the two matrices have high values in the same positions, the dot product's output will be high. If they don't, it will be low. In this way, a single value -- the output of the dot product -- can tell us whether the pixel pattern in the underlying image matches the pixel pattern expressed by our filter. 

Let's imagine that our filter expresses a horizontal line, with high values along its second row and low values in the first and third rows. Now picture that we start in the upper lefthand corner of the underlying image, and we move the filter across the image step by step until it reaches the upper righthand corner. The size of the step is known as *stride*. You can move the filter to the right one column at a time, or you can choose to make larger steps. 

At each step, you take another dot product, and you place the results of that dot product in a third matrix known as an *activation map*. The width, or number of columns, of the activation map is equal to the number of steps the filter takes to traverse the underlying image. Since larger strides lead to fewer steps, a big stride will produce a smaller activation map. This is important, because the size of the matrices that convolutional networks process and produce at each layer is directly proportional to how computationally expensive they are and how much time they take to train. A larger stride means less time and compute.

A filter superimposed on the first three rows will slide across them and then begin again with rows 4-6 of the same image. If it has a stride of three, then it will produce a matrix of dot products that is 10x10. That same filter representing a horizontal line can be applied to all three channels of the underlying image, R, G and B. And the three 10x10 activation maps can be added together, so that the aggregate activation map for a horizontal line on all three channels of the underlying image is also 10x10.

Now, because images have lines going in many directions, and contain many different kinds of shapes and pixel patterns, you will want to slide other filters across the underlying image in search of those patterns. You could, for example, look for 96 different patterns in the pixels. Those 96 patterns will create a stack of 96 activation maps, resulting in a new volume that is 10x10x96. In the diagram below, we've relabeled the input image, the kernels and the output activation maps to make sure we're clear. 

![Alt text](./img/karpathy-convnet-labels.png) 

What we just described is a convolution. You can think of Convolution as a fancy kind of multiplication used in signal processing. Another way to think about the two matrices creating a dot product is as two functions. The image is the underlying function, and the filter is the function you roll over it. 

<iframe src="http://mathworld.wolfram.com//images/gifs/convgaus.gif" width="100%" height="250px;" style="border:none;"></iframe>

One of the main problems with images is that they are high-dimensional, which means they cost a lot of time and computing power to process. Convolutional networks are designed to reduce the dimensionality of images in a variety of ways. Filter stride is one way to reduce dimensionality. Another way is through downsampling. 

## <a name="max">Max Pooling/Downsampling</a>

The next layer in a convolutional network has three names: max pooling, downsampling and subsampling. The activation maps are fed into a downsampling layer, and like convolutions, this method is applied one patch at a time. In this case, max pooling simply takes the largest value from one patch of an image, places it in a new matrix next to the max values from other patches, and discards the rest of the information contained in the activation maps.

![Alt text](./img/maxpool.png) 
*Credit to [Andrej Karpathy](https://cs231n.github.io/).*

Only the locations on the image that showed the strongest correlation to each feature (the maximum value) are preserved, and those maximum values combine to form a lower-dimensional space. 

Much information about lesser values is lost in this step, which has spurred research into alternative methods. But downsampling has the advantage, precisely because information is lost, of decreasing the amount of storage and processing required. 

### Alternating Layers

The image below is another attempt to show the sequence of transformations involved in a typical convolutional network. 

![Alt text](./img/convnet.png) 

From left to right you see:

* The actual input image that is scanned for features. The light rectangle is the filter that passes over it. 
* Activation maps stacked atop one another, one for each filter you employ. The larger rectangle is one patch to be downsampled.  
* The activation maps condensed through downsampling.
* A new set of activation maps created by passing filters over the first downsampled stack. 
* The second downsampling, which condenses the second set of activation maps. 
* A fully connected layer that classifies output with one label per node. 

As more and more information is lost, the patterns processed by the convolutional net become more abstract and grow more distant from visual patterns we recognize as humans. So forgive yourself, and us, if convolutional networks do not offer easy intuitions as they grow deeper. 

## <a name="code">DL4J Code Example</a>

Here's one example of how you might configure a ConvNet with Deeplearning4j:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/LenetMnistExample.java?slice=43:87"></script>

All Deeplearning4j [examples of convolutional networks are available here](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution).

### <a name="beginner">Other Deeplearning4j Tutorials</a>
* [Introduction to Neural Networks](./neuralnet-overview)
* [LSTMs and Recurrent Networks](./lstm)
* [Word2vec](./word2vec)
* [Restricted Boltzmann Machines](./restrictedboltzmannmachine)
* [Eigenvectors, Covariance, PCA and Entropy](./eigenvector)
* [Neural Networks and Regression](./linear-regression)

## <a name="resource">Other Resources</a>

To see DL4J convolutional networks in action, please run our [examples](https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/) after following the instructions on the [Quickstart page](./quickstart). 

## Resources Table of Contents
- [Papers](#papers)
  - [ImageNet Classification](#imagenet-classification)
  - [Object Detection](#object-detection)
  - [Object Tracking](#object-tracking)
  - [Low-Level Vision](#low-level-vision)
    - [Super-Resolution](#super-resolution)
    - [Other Applications](#other-applications)
  - [Edge Detection](#edge-detection)
  - [Semantic Segmentation](#semantic-segmentation)
  - [Visual Attention and Saliency](#visual-attention-and-saliency)
  - [Object Recognition](#object-recognition)
  - [Human Pose Estimation](#human-pose-estimation)
  - [Understanding CNN](#understanding-cnn)
  - [Image and Language](#image-and-language)
    - [Image Captioning](#image-captioning)
    - [Video Captioning](#video-captioning)
    - [Question Answering](#question-answering)
  - [Image Generation](#image-generation)
  - [Other Topics](#other-topics)
- [Courses](#courses)
- [Books](#books)
- [Videos](#videos)
- [Software](#software)
  - [Framework](#framework)
  - [Applications](#applications)
- [Tutorials](#tutorials)
- [Blogs](#blogs)

## Papers

### ImageNet Classification
![classification](https://cloud.githubusercontent.com/assets/5226447/8451949/327b9566-2022-11e5-8b34-53b4a64c13ad.PNG)
(from Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, ImageNet Classification with Deep Convolutional Neural Networks, NIPS, 2012.)
* Microsoft (Deep Residual Learning) [[Paper](http://arxiv.org/pdf/1512.03385v1.pdf)][[Slide](http://image-net.org/challenges/talks/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)]
  * Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Deep Residual Learning for Image Recognition, arXiv:1512.03385.
* Microsoft (PReLu/Weight Initialization) [[Paper]](http://arxiv.org/pdf/1502.01852)
  * Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, arXiv:1502.01852.
* Batch Normalization [[Paper]](http://arxiv.org/pdf/1502.03167)
  * Sergey Ioffe, Christian Szegedy, Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, arXiv:1502.03167.
* GoogLeNet [[Paper]](http://arxiv.org/pdf/1409.4842)
  * Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich, CVPR, 2015.
* VGG-Net [[Web]](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) [[Paper]](http://arxiv.org/pdf/1409.1556)
  * Karen Simonyan and Andrew Zisserman, Very Deep Convolutional Networks for Large-Scale Visual Recognition, ICLR, 2015.
* AlexNet [[Paper]](http://papers.nips.cc/book/advances-in-neural-information-processing-systems-25-2012)
  * Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, ImageNet Classification with Deep Convolutional Neural Networks, NIPS, 2012.

### Object Detection
![object_detection](https://cloud.githubusercontent.com/assets/5226447/8452063/f76ba500-2022-11e5-8db1-2cd5d490e3b3.PNG)
(from Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, arXiv:1506.01497.)

* PVANET [[Paper]](https://arxiv.org/pdf/1608.08021) [[Code]](https://github.com/sanghoon/pva-faster-rcnn)
  * Kye-Hyeon Kim, Sanghoon Hong, Byungseok Roh, Yeongjae Cheon, Minje Park, PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection, arXiv:1608.08021
* OverFeat, NYU [[Paper]](http://arxiv.org/pdf/1312.6229.pdf)
  * OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks, ICLR, 2014.
* R-CNN, UC Berkeley [[Paper-CVPR14]](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) [[Paper-arXiv14]](http://arxiv.org/pdf/1311.2524)
  * Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik, Rich feature hierarchies for accurate object detection and semantic segmentation, CVPR, 2014.
* SPP, Microsoft Research [[Paper]](http://arxiv.org/pdf/1406.4729)
  * Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition, ECCV, 2014.
* Fast R-CNN, Microsoft Research [[Paper]](http://arxiv.org/pdf/1504.08083)
  * Ross Girshick, Fast R-CNN, arXiv:1504.08083.
* Faster R-CNN, Microsoft Research [[Paper]](http://arxiv.org/pdf/1506.01497)
  * Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun, Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks, arXiv:1506.01497.
* R-CNN minus R, Oxford [[Paper]](http://arxiv.org/pdf/1506.06981)
  * Karel Lenc, Andrea Vedaldi, R-CNN minus R, arXiv:1506.06981.
* End-to-end people detection in crowded scenes [[Paper]](http://arxiv.org/abs/1506.04878)
  * Russell Stewart, Mykhaylo Andriluka, End-to-end people detection in crowded scenes, arXiv:1506.04878.
* You Only Look Once: Unified, Real-Time Object Detection [[Paper]](http://arxiv.org/abs/1506.02640), [[Paper Version 2]](https://arxiv.org/abs/1612.08242), [[C Code]](https://github.com/pjreddie/darknet), [[Tensorflow Code]](https://github.com/thtrieu/darkflow)
  * Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi, You Only Look Once: Unified, Real-Time Object Detection, arXiv:1506.02640
  * Joseph Redmon, Ali Farhadi (Version 2)
* Inside-Outside Net [[Paper]](http://arxiv.org/abs/1512.04143)
  * Sean Bell, C. Lawrence Zitnick, Kavita Bala, Ross Girshick, Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks
* Deep Residual Network (Current State-of-the-Art) [[Paper]](http://arxiv.org/abs/1512.03385)
  * Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, Deep Residual Learning for Image Recognition
* Weakly Supervised Object Localization with Multi-fold Multiple Instance Learning [[Paper](http://arxiv.org/pdf/1503.00949.pdf)]
* R-FCN [[Paper]](https://arxiv.org/abs/1605.06409) [[Code]](https://github.com/daijifeng001/R-FCN)
  * Jifeng Dai, Yi Li, Kaiming He, Jian Sun, R-FCN: Object Detection via Region-based Fully Convolutional Networks
* SSD [[Paper]](https://arxiv.org/pdf/1512.02325v2.pdf) [[Code]](https://github.com/weiliu89/caffe/tree/ssd)
  * Wei Liu1, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg, SSD: Single Shot MultiBox Detector, arXiv:1512.02325
* Speed/accuracy trade-offs for modern convolutional object detectors [[Paper]](https://arxiv.org/pdf/1611.10012v1.pdf)
  * Jonathan Huang, Vivek Rathod, Chen Sun, Menglong Zhu, Anoop Korattikara, Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, Kevin Murphy, Google Research, arXiv:1611.10012

### Video Classification
* Nicolas Ballas, Li Yao, Pal Chris, Aaron Courville, "Delving Deeper into Convolutional Networks for Learning Video Representations", ICLR 2016. [[Paper](http://arxiv.org/pdf/1511.06432v4.pdf)]
* Michael Mathieu, camille couprie, Yann Lecun, "Deep Multi Scale Video Prediction Beyond Mean Square Error", ICLR 2016. [[Paper](http://arxiv.org/pdf/1511.05440v6.pdf)]

### Object Tracking
* Seunghoon Hong, Tackgeun You, Suha Kwak, Bohyung Han, Online Tracking by Learning Discriminative Saliency Map with Convolutional Neural Network, arXiv:1502.06796. [[Paper]](http://arxiv.org/pdf/1502.06796)
* Hanxi Li, Yi Li and Fatih Porikli, DeepTrack: Learning Discriminative Feature Representations by Convolutional Neural Networks for Visual Tracking, BMVC, 2014. [[Paper]](http://www.bmva.org/bmvc/2014/files/paper028.pdf)
* N Wang, DY Yeung, Learning a Deep Compact Image Representation for Visual Tracking, NIPS, 2013. [[Paper]](http://winsty.net/papers/dlt.pdf)
* Chao Ma, Jia-Bin Huang, Xiaokang Yang and Ming-Hsuan Yang, Hierarchical Convolutional Features for Visual Tracking, ICCV 2015 [[Paper](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Ma_Hierarchical_Convolutional_Features_ICCV_2015_paper.pdf)] [[Code](https://github.com/jbhuang0604/CF2)]
* Lijun Wang, Wanli Ouyang, Xiaogang Wang, and Huchuan Lu, Visual Tracking with fully Convolutional Networks, ICCV 2015  [[Paper](http://202.118.75.4/lu/Paper/ICCV2015/iccv15_lijun.pdf)] [[Code](https://github.com/scott89/FCNT)]
* Hyeonseob Namand Bohyung Han, Learning Multi-Domain Convolutional Neural Networks for Visual Tracking, [[Paper](http://arxiv.org/pdf/1510.07945.pdf)] [[Code](https://github.com/HyeonseobNam/MDNet)] [[Project Page](http://cvlab.postech.ac.kr/research/mdnet/)]

### Low-Level Vision

#### Super-Resolution
* Iterative Image Reconstruction
  * Sven Behnke: Learning Iterative Image Reconstruction. IJCAI, 2001. [[Paper]](http://www.ais.uni-bonn.de/behnke/papers/ijcai01.pdf)
  * Sven Behnke: Learning Iterative Image Reconstruction in the Neural Abstraction Pyramid. International Journal of Computational Intelligence and Applications, vol. 1, no. 4, pp. 427-438, 2001. [[Paper]](http://www.ais.uni-bonn.de/behnke/papers/ijcia01.pdf)
* Super-Resolution (SRCNN) [[Web]](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html) [[Paper-ECCV14]](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepresolution.pdf) [[Paper-arXiv15]](http://arxiv.org/pdf/1501.00092.pdf)
  * Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, Learning a Deep Convolutional Network for Image Super-Resolution, ECCV, 2014.
  * Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang, Image Super-Resolution Using Deep Convolutional Networks, arXiv:1501.00092.
* Very Deep Super-Resolution
  * Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee, Accurate Image Super-Resolution Using Very Deep Convolutional Networks, arXiv:1511.04587, 2015. [[Paper]](http://arxiv.org/abs/1511.04587)
* Deeply-Recursive Convolutional Network
  * Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee, Deeply-Recursive Convolutional Network for Image Super-Resolution, arXiv:1511.04491, 2015. [[Paper]](http://arxiv.org/abs/1511.04491)
* Casade-Sparse-Coding-Network
  * Zhaowen Wang, Ding Liu, Wei Han, Jianchao Yang and Thomas S. Huang, Deep Networks for Image Super-Resolution with Sparse Prior. ICCV, 2015. [[Paper]](http://www.ifp.illinois.edu/~dingliu2/iccv15/iccv15.pdf) [[Code]](http://www.ifp.illinois.edu/~dingliu2/iccv15/)
* Perceptual Losses for Super-Resolution
  * Justin Johnson, Alexandre Alahi, Li Fei-Fei, Perceptual Losses for Real-Time Style Transfer and Super-Resolution, arXiv:1603.08155, 2016. [[Paper]](http://arxiv.org/abs/1603.08155) [[Supplementary]](http://cs.stanford.edu/people/jcjohns/papers/fast-style/fast-style-supp.pdf)
* SRGAN
  * Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi, Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, arXiv:1609.04802v3, 2016. [[Paper]](https://arxiv.org/pdf/1609.04802v3.pdf)
* Others
  * Osendorfer, Christian, Hubert Soyer, and Patrick van der Smagt, Image Super-Resolution with Fast Approximate Convolutional Sparse Coding, ICONIP, 2014. [[Paper ICONIP-2014]](http://brml.org/uploads/tx_sibibtex/281.pdf)

#### Other Applications
* Optical Flow (FlowNet) [[Paper]](http://arxiv.org/pdf/1504.06852)
  * Philipp Fischer, Alexey Dosovitskiy, Eddy Ilg, Philip Häusser, Caner Hazırbaş, Vladimir Golkov, Patrick van der Smagt, Daniel Cremers, Thomas Brox, FlowNet: Learning Optical Flow with Convolutional Networks, arXiv:1504.06852.
* Compression Artifacts Reduction [[Paper-arXiv15]](http://arxiv.org/pdf/1504.06993)
  * Chao Dong, Yubin Deng, Chen Change Loy, Xiaoou Tang, Compression Artifacts Reduction by a Deep Convolutional Network, arXiv:1504.06993.
* Blur Removal
  * Christian J. Schuler, Michael Hirsch, Stefan Harmeling, Bernhard Schölkopf, Learning to Deblur, arXiv:1406.7444 [[Paper]](http://arxiv.org/pdf/1406.7444.pdf)
  * Jian Sun, Wenfei Cao, Zongben Xu, Jean Ponce, Learning a Convolutional Neural Network for Non-uniform Motion Blur Removal, CVPR, 2015 [[Paper]](http://arxiv.org/pdf/1503.00593)
* Image Deconvolution [[Web]](http://lxu.me/projects/dcnn/) [[Paper]](http://lxu.me/mypapers/dcnn_nips14.pdf)
  * Li Xu, Jimmy SJ. Ren, Ce Liu, Jiaya Jia, Deep Convolutional Neural Network for Image Deconvolution, NIPS, 2014.
* Deep Edge-Aware Filter [[Paper]](http://jmlr.org/proceedings/papers/v37/xub15.pdf)
  * Li Xu, Jimmy SJ. Ren, Qiong Yan, Renjie Liao, Jiaya Jia, Deep Edge-Aware Filters, ICML, 2015.
* Computing the Stereo Matching Cost with a Convolutional Neural Network [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zbontar_Computing_the_Stereo_2015_CVPR_paper.pdf)
  * Jure Žbontar, Yann LeCun, Computing the Stereo Matching Cost with a Convolutional Neural Network, CVPR, 2015.
* Colorful Image Colorization Richard Zhang, Phillip Isola, Alexei A. Efros, ECCV, 2016 [[Paper]](http://arxiv.org/pdf/1603.08511.pdf), [[Code]](https://github.com/richzhang/colorization)
* Ryan Dahl, [[Blog]](http://tinyclouds.org/colorize/)
* Feature Learning by Inpainting[[Paper]](https://arxiv.org/pdf/1604.07379v1.pdf)[[Code]](https://github.com/pathak22/context-encoder)
  * Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, Alexei A. Efros, Context Encoders: Feature Learning by Inpainting, CVPR, 2016

### Edge Detection
![edge_detection](https://cloud.githubusercontent.com/assets/5226447/8452371/93ca6f7e-2025-11e5-90f2-d428fd5ff7ac.PNG)
(from Gedas Bertasius, Jianbo Shi, Lorenzo Torresani, DeepEdge: A Multi-Scale Bifurcated Deep Network for Top-Down Contour Detection, CVPR, 2015.)

* Holistically-Nested Edge Detection [[Paper]](http://arxiv.org/pdf/1504.06375) [[Code]](https://github.com/s9xie/hed)
  * Saining Xie, Zhuowen Tu, Holistically-Nested Edge Detection, arXiv:1504.06375.
* DeepEdge [[Paper]](http://arxiv.org/pdf/1412.1123)
  * Gedas Bertasius, Jianbo Shi, Lorenzo Torresani, DeepEdge: A Multi-Scale Bifurcated Deep Network for Top-Down Contour Detection, CVPR, 2015.
* DeepContour [[Paper]](http://mc.eistar.net/UpLoadFiles/Papers/DeepContour_cvpr15.pdf)
  * Wei Shen, Xinggang Wang, Yan Wang, Xiang Bai, Zhijiang Zhang, DeepContour: A Deep Convolutional Feature Learned by Positive-Sharing Loss for Contour Detection, CVPR, 2015.

### Semantic Segmentation
![semantic_segmantation](https://cloud.githubusercontent.com/assets/5226447/8452076/0ba8340c-2023-11e5-88bc-bebf4509b6bb.PNG)
(from Jifeng Dai, Kaiming He, Jian Sun, BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation, arXiv:1503.01640.)
* PASCAL VOC2012 Challenge Leaderboard (01 Sep. 2016)
  ![VOC2012_top_rankings](https://cloud.githubusercontent.com/assets/3803777/18164608/c3678488-7038-11e6-9ec1-74a1542dce13.png)
  (from PASCAL VOC2012 [leaderboards](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6))
* SEC: Seed, Expand and Constrain
  *  Alexander Kolesnikov, Christoph Lampert, Seed, Expand and Constrain: Three Principles for Weakly-Supervised Image Segmentation, ECCV, 2016. [[Paper]](http://pub.ist.ac.at/~akolesnikov/files/ECCV2016/main.pdf) [[Code]](https://github.com/kolesman/SEC)
* Adelaide
  * Guosheng Lin, Chunhua Shen, Ian Reid, Anton van dan Hengel, Efficient piecewise training of deep structured models for semantic segmentation, arXiv:1504.01013. [[Paper]](http://arxiv.org/pdf/1504.01013) (1st ranked in VOC2012)
  * Guosheng Lin, Chunhua Shen, Ian Reid, Anton van den Hengel, Deeply Learning the Messages in Message Passing Inference, arXiv:1508.02108. [[Paper]](http://arxiv.org/pdf/1506.02108) (4th ranked in VOC2012)
* Deep Parsing Network (DPN)
  * Ziwei Liu, Xiaoxiao Li, Ping Luo, Chen Change Loy, Xiaoou Tang, Semantic Image Segmentation via Deep Parsing Network, arXiv:1509.02634 / ICCV 2015 [[Paper]](http://arxiv.org/pdf/1509.02634.pdf) (2nd ranked in VOC 2012)
* CentraleSuperBoundaries, INRIA [[Paper]](http://arxiv.org/pdf/1511.07386)
  * Iasonas Kokkinos, Surpassing Humans in Boundary Detection using Deep Learning, arXiv:1411.07386 (4th ranked in VOC 2012)
* BoxSup [[Paper]](http://arxiv.org/pdf/1503.01640)
  * Jifeng Dai, Kaiming He, Jian Sun, BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation, arXiv:1503.01640. (6th ranked in VOC2012)
* POSTECH
  * Hyeonwoo Noh, Seunghoon Hong, Bohyung Han, Learning Deconvolution Network for Semantic Segmentation, arXiv:1505.04366. [[Paper]](http://arxiv.org/pdf/1505.04366) (7th ranked in VOC2012)
  * Seunghoon Hong, Hyeonwoo Noh, Bohyung Han, Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation, arXiv:1506.04924. [[Paper]](http://arxiv.org/pdf/1506.04924)
  * Seunghoon Hong,Junhyuk Oh,	Bohyung Han, and	Honglak Lee, Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network, arXiv:1512.07928 [[Paper](http://arxiv.org/pdf/1512.07928.pdf)] [[Project Page](http://cvlab.postech.ac.kr/research/transfernet/)]
* Conditional Random Fields as Recurrent Neural Networks [[Paper]](http://arxiv.org/pdf/1502.03240)
  * Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du, Chang Huang, Philip H. S. Torr, Conditional Random Fields as Recurrent Neural Networks, arXiv:1502.03240. (8th ranked in VOC2012)
* DeepLab
  * Liang-Chieh Chen, George Papandreou, Kevin Murphy, Alan L. Yuille, Weakly-and semi-supervised learning of a DCNN for semantic image segmentation, arXiv:1502.02734. [[Paper]](http://arxiv.org/pdf/1502.02734) (9th ranked in VOC2012)
* Zoom-out [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mostajabi_Feedforward_Semantic_Segmentation_2015_CVPR_paper.pdf)
  * Mohammadreza Mostajabi, Payman Yadollahpour, Gregory Shakhnarovich, Feedforward Semantic Segmentation With Zoom-Out Features, CVPR, 2015
* Joint Calibration [[Paper]](http://arxiv.org/pdf/1507.01581)
  * Holger Caesar, Jasper Uijlings, Vittorio Ferrari, Joint Calibration for Semantic Segmentation, arXiv:1507.01581.
* Fully Convolutional Networks for Semantic Segmentation [[Paper-CVPR15]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) [[Paper-arXiv15]](http://arxiv.org/pdf/1411.4038)
  * Jonathan Long, Evan Shelhamer, Trevor Darrell, Fully Convolutional Networks for Semantic Segmentation, CVPR, 2015.
* Hypercolumn [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hariharan_Hypercolumns_for_Object_2015_CVPR_paper.pdf)
  * Bharath Hariharan, Pablo Arbelaez, Ross Girshick, Jitendra Malik, Hypercolumns for Object Segmentation and Fine-Grained Localization, CVPR, 2015.
* Deep Hierarchical Parsing
  * Abhishek Sharma, Oncel Tuzel, David W. Jacobs, Deep Hierarchical Parsing for Semantic Segmentation, CVPR, 2015. [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sharma_Deep_Hierarchical_Parsing_2015_CVPR_paper.pdf)
* Learning Hierarchical Features for Scene Labeling [[Paper-ICML12]](http://yann.lecun.com/exdb/publis/pdf/farabet-icml-12.pdf) [[Paper-PAMI13]](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)
  * Clement Farabet, Camille Couprie, Laurent Najman, Yann LeCun, Scene Parsing with Multiscale Feature Learning, Purity Trees, and Optimal Covers, ICML, 2012.
  * Clement Farabet, Camille Couprie, Laurent Najman, Yann LeCun, Learning Hierarchical Features for Scene Labeling, PAMI, 2013.
* University of Cambridge [[Web]](http://mi.eng.cam.ac.uk/projects/segnet/)
  * Vijay Badrinarayanan, Alex Kendall and Roberto Cipolla "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation." arXiv preprint arXiv:1511.00561, 2015. [[Paper]](http://arxiv.org/abs/1511.00561)
* Alex Kendall, Vijay Badrinarayanan and Roberto Cipolla "Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding." arXiv preprint arXiv:1511.02680, 2015. [[Paper]](http://arxiv.org/abs/1511.00561)
* Princeton
  * Fisher Yu, Vladlen Koltun, "Multi-Scale Context Aggregation by Dilated Convolutions", ICLR 2016, [[Paper](http://arxiv.org/pdf/1511.07122v2.pdf)]
* Univ. of Washington, Allen AI
  * Hamid Izadinia, Fereshteh Sadeghi, Santosh Kumar Divvala, Yejin Choi, Ali Farhadi, "Segment-Phrase Table for Semantic Segmentation, Visual Entailment and Paraphrasing", ICCV, 2015, [[Paper](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Izadinia_Segment-Phrase_Table_for_ICCV_2015_paper.pdf)]
* INRIA
  * Iasonas Kokkinos, "Pusing the Boundaries of Boundary Detection Using deep Learning", ICLR 2016, [[Paper](http://arxiv.org/pdf/1511.07386v2.pdf)]
* UCSB
  * Niloufar Pourian, S. Karthikeyan, and B.S. Manjunath, "Weakly supervised graph based semantic segmentation by learning communities of image-parts", ICCV, 2015, [[Paper](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Pourian_Weakly_Supervised_Graph_ICCV_2015_paper.pdf)]

### Visual Attention and Saliency
![saliency](https://cloud.githubusercontent.com/assets/5226447/8492362/7ec65b88-2183-11e5-978f-017e45ddba32.png)
(from Nian Liu, Junwei Han, Dingwen Zhang, Shifeng Wen, Tianming Liu, Predicting Eye Fixations using Convolutional Neural Networks, CVPR, 2015.)

* Mr-CNN [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Predicting_Eye_Fixations_2015_CVPR_paper.pdf)
  * Nian Liu, Junwei Han, Dingwen Zhang, Shifeng Wen, Tianming Liu, Predicting Eye Fixations using Convolutional Neural Networks, CVPR, 2015.
* Learning a Sequential Search for Landmarks [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Singh_Learning_a_Sequential_2015_CVPR_paper.pdf)
  * Saurabh Singh, Derek Hoiem, David Forsyth, Learning a Sequential Search for Landmarks, CVPR, 2015.
* Multiple Object Recognition with Visual Attention [[Paper]](http://arxiv.org/pdf/1412.7755.pdf)
  * Jimmy Lei Ba, Volodymyr Mnih, Koray Kavukcuoglu, Multiple Object Recognition with Visual Attention, ICLR, 2015.
* Recurrent Models of Visual Attention [[Paper]](http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf)
  * Volodymyr Mnih, Nicolas Heess, Alex Graves, Koray Kavukcuoglu, Recurrent Models of Visual Attention, NIPS, 2014.

### Object Recognition
* Weakly-supervised learning with convolutional neural networks [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Oquab_Is_Object_Localization_2015_CVPR_paper.pdf)
  * Maxime Oquab, Leon Bottou, Ivan Laptev, Josef Sivic, Is object localization for free? – Weakly-supervised learning with convolutional neural networks, CVPR, 2015.
* FV-CNN [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Cimpoi_Deep_Filter_Banks_2015_CVPR_paper.pdf)
  * Mircea Cimpoi, Subhransu Maji, Andrea Vedaldi, Deep Filter Banks for Texture Recognition and Segmentation, CVPR, 2015.

### Human Pose Estimation
* Zhe Cao, Tomas Simon, Shih-En Wei, and Yaser Sheikh, Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields, CVPR, 2017.
* Leonid Pishchulin, Eldar Insafutdinov, Siyu Tang, Bjoern Andres, Mykhaylo Andriluka, Peter Gehler, and Bernt Schiele, Deepcut: Joint subset partition and labeling for multi person pose estimation, CVPR, 2016.
* Shih-En Wei, Varun Ramakrishna, Takeo Kanade, and Yaser Sheikh, Convolutional pose machines, CVPR, 2016.
* Alejandro Newell, Kaiyu Yang, and Jia Deng, Stacked hourglass networks for human pose estimation, ECCV, 2016.
* Tomas Pfister, James Charles, and Andrew Zisserman, Flowing convnets for human pose estimation in videos, ICCV, 2015.
* Jonathan J. Tompson, Arjun Jain, Yann LeCun, Christoph Bregler, Joint training of a convolutional network and a graphical model for human pose estimation, NIPS, 2014.

### Understanding CNN
![understanding](https://cloud.githubusercontent.com/assets/5226447/8452083/1aaa0066-2023-11e5-800b-2248ead51584.PNG)
(from Aravindh Mahendran, Andrea Vedaldi, Understanding Deep Image Representations by Inverting Them, CVPR, 2015.)

* Karel Lenc, Andrea Vedaldi, Understanding image representations by measuring their equivariance and equivalence, CVPR, 2015. [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Lenc_Understanding_Image_Representations_2015_CVPR_paper.pdf)
* Anh Nguyen, Jason Yosinski, Jeff Clune, Deep Neural Networks are Easily Fooled:High Confidence Predictions for Unrecognizable Images, CVPR, 2015. [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Nguyen_Deep_Neural_Networks_2015_CVPR_paper.pdf)
* Aravindh Mahendran, Andrea Vedaldi, Understanding Deep Image Representations by Inverting Them, CVPR, 2015. [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mahendran_Understanding_Deep_Image_2015_CVPR_paper.pdf)
* Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, Antonio Torralba, Object Detectors Emerge in Deep Scene CNNs, ICLR, 2015. [[arXiv Paper]](http://arxiv.org/abs/1412.6856)
* Alexey Dosovitskiy, Thomas Brox, Inverting Visual Representations with Convolutional Networks, arXiv, 2015. [[Paper]](http://arxiv.org/abs/1506.02753)
* Matthrew Zeiler, Rob Fergus, Visualizing and Understanding Convolutional Networks, ECCV, 2014. [[Paper]](https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)


### Image and Language

#### Image Captioning
![image_captioning](https://cloud.githubusercontent.com/assets/5226447/8452051/e8f81030-2022-11e5-85db-c68e7d8251ce.PNG)
(from Andrej Karpathy, Li Fei-Fei, Deep Visual-Semantic Alignments for Generating Image Description, CVPR, 2015.)

* UCLA / Baidu [[Paper]](http://arxiv.org/pdf/1410.1090)
  * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Alan L. Yuille, Explain Images with Multimodal Recurrent Neural Networks, arXiv:1410.1090.
* Toronto [[Paper]](http://arxiv.org/pdf/1411.2539)
  * Ryan Kiros, Ruslan Salakhutdinov, Richard S. Zemel, Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models, arXiv:1411.2539.
* Berkeley [[Paper]](http://arxiv.org/pdf/1411.4389)
  * Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, Trevor Darrell, Long-term Recurrent Convolutional Networks for Visual Recognition and Description, arXiv:1411.4389.
* Google [[Paper]](http://arxiv.org/pdf/1411.4555)
  * Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan, Show and Tell: A Neural Image Caption Generator, arXiv:1411.4555.
* Stanford [[Web]](http://cs.stanford.edu/people/karpathy/deepimagesent/) [[Paper]](http://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
  * Andrej Karpathy, Li Fei-Fei, Deep Visual-Semantic Alignments for Generating Image Description, CVPR, 2015.
* UML / UT [[Paper]](http://arxiv.org/pdf/1412.4729)
  * Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond Mooney, Kate Saenko, Translating Videos to Natural Language Using Deep Recurrent Neural Networks, NAACL-HLT, 2015.
* CMU / Microsoft [[Paper-arXiv]](http://arxiv.org/pdf/1411.5654) [[Paper-CVPR]](http://www.cs.cmu.edu/~xinleic/papers/cvpr15_rnn.pdf)
  * Xinlei Chen, C. Lawrence Zitnick, Learning a Recurrent Visual Representation for Image Caption Generation, arXiv:1411.5654.
  * Xinlei Chen, C. Lawrence Zitnick, Mind’s Eye: A Recurrent Visual Representation for Image Caption Generation, CVPR 2015
* Microsoft [[Paper]](http://arxiv.org/pdf/1411.4952)
  * Hao Fang, Saurabh Gupta, Forrest Iandola, Rupesh Srivastava, Li Deng, Piotr Dollár, Jianfeng Gao, Xiaodong He, Margaret Mitchell, John C. Platt, C. Lawrence Zitnick, Geoffrey Zweig, From Captions to Visual Concepts and Back, CVPR, 2015.
* Univ. Montreal / Univ. Toronto [[Web](http://kelvinxu.github.io/projects/capgen.html)] [[Paper](http://www.cs.toronto.edu/~zemel/documents/captionAttn.pdf)]
  * Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard S. Zemel, Yoshua Bengio, Show, Attend, and Tell: Neural Image Caption Generation with Visual Attention, arXiv:1502.03044 / ICML 2015
* Idiap / EPFL / Facebook [[Paper](http://arxiv.org/pdf/1502.03671)]
  * Remi Lebret, Pedro O. Pinheiro, Ronan Collobert, Phrase-based Image Captioning, arXiv:1502.03671 / ICML 2015
* UCLA / Baidu [[Paper](http://arxiv.org/pdf/1504.06692)]
  * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang, Alan L. Yuille, Learning like a Child: Fast Novel Visual Concept Learning from Sentence Descriptions of Images, arXiv:1504.06692
* MS + Berkeley
  * Jacob Devlin, Saurabh Gupta, Ross Girshick, Margaret Mitchell, C. Lawrence Zitnick, Exploring Nearest Neighbor Approaches for Image Captioning, arXiv:1505.04467 [[Paper](http://arxiv.org/pdf/1505.04467.pdf)]
  * Jacob Devlin, Hao Cheng, Hao Fang, Saurabh Gupta, Li Deng, Xiaodong He, Geoffrey Zweig, Margaret Mitchell, Language Models for Image Captioning: The Quirks and What Works, arXiv:1505.01809 [[Paper](http://arxiv.org/pdf/1505.01809.pdf)]
* Adelaide [[Paper](http://arxiv.org/pdf/1506.01144.pdf)]
  * Qi Wu, Chunhua Shen, Anton van den Hengel, Lingqiao Liu, Anthony Dick, Image Captioning with an Intermediate Attributes Layer, arXiv:1506.01144
* Tilburg [[Paper](http://arxiv.org/pdf/1506.03694.pdf)]
  * Grzegorz Chrupala, Akos Kadar, Afra Alishahi, Learning language through pictures, arXiv:1506.03694
* Univ. Montreal [[Paper](http://arxiv.org/pdf/1507.01053.pdf)]
  * Kyunghyun Cho, Aaron Courville, Yoshua Bengio, Describing Multimedia Content using Attention-based Encoder-Decoder Networks, arXiv:1507.01053
* Cornell [[Paper](http://arxiv.org/pdf/1508.02091.pdf)]
  * Jack Hessel, Nicolas Savva, Michael J. Wilber, Image Representations and New Domains in Neural Image Captioning, arXiv:1508.02091
* MS + City Univ. of HongKong [[Paper](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yao_Learning_Query_and_ICCV_2015_paper.pdf)]
  * Ting Yao, Tao Mei, and Chong-Wah Ngo, "Learning Query and Image Similarities
    with Ranking Canonical Correlation Analysis", ICCV, 2015

#### Video Captioning
* Berkeley [[Web]](http://jeffdonahue.com/lrcn/) [[Paper]](http://arxiv.org/pdf/1411.4389.pdf)
  * Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, Trevor Darrell, Long-term Recurrent Convolutional Networks for Visual Recognition and Description, CVPR, 2015.
* UT / UML / Berkeley [[Paper]](http://arxiv.org/pdf/1412.4729)
  * Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond Mooney, Kate Saenko, Translating Videos to Natural Language Using Deep Recurrent Neural Networks, arXiv:1412.4729.
* Microsoft [[Paper]](http://arxiv.org/pdf/1505.01861)
  * Yingwei Pan, Tao Mei, Ting Yao, Houqiang Li, Yong Rui, Joint Modeling Embedding and Translation to Bridge Video and Language, arXiv:1505.01861.
* UT / Berkeley / UML [[Paper]](http://arxiv.org/pdf/1505.00487)
  * Subhashini Venugopalan, Marcus Rohrbach, Jeff Donahue, Raymond Mooney, Trevor Darrell, Kate Saenko, Sequence to Sequence--Video to Text, arXiv:1505.00487.
* Univ. Montreal / Univ. Sherbrooke [[Paper](http://arxiv.org/pdf/1502.08029.pdf)]
  * Li Yao, Atousa Torabi, Kyunghyun Cho, Nicolas Ballas, Christopher Pal, Hugo Larochelle, Aaron Courville, Describing Videos by Exploiting Temporal Structure, arXiv:1502.08029
* MPI / Berkeley [[Paper](http://arxiv.org/pdf/1506.01698.pdf)]
  * Anna Rohrbach, Marcus Rohrbach, Bernt Schiele, The Long-Short Story of Movie Description, arXiv:1506.01698
* Univ. Toronto / MIT [[Paper](http://arxiv.org/pdf/1506.06724.pdf)]
  * Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, Sanja Fidler, Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books, arXiv:1506.06724
* Univ. Montreal [[Paper](http://arxiv.org/pdf/1507.01053.pdf)]
  * Kyunghyun Cho, Aaron Courville, Yoshua Bengio, Describing Multimedia Content using Attention-based Encoder-Decoder Networks, arXiv:1507.01053
* TAU / USC [[paper](https://arxiv.org/pdf/1612.06950.pdf)]
  * Dotan Kaufman, Gil Levi, Tal Hassner, Lior Wolf, Temporal Tessellation for Video Annotation and Summarization, arXiv:1612.06950.

#### Question Answering
![question_answering](https://cloud.githubusercontent.com/assets/5226447/8452068/ffe7b1f6-2022-11e5-87ab-4f6d4696c220.PNG)
(from Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, Devi Parikh, VQA: Visual Question Answering, CVPR, 2015 SUNw:Scene Understanding workshop)

* Virginia Tech / MSR [[Web]](http://www.visualqa.org/) [[Paper]](http://arxiv.org/pdf/1505.00468)
  * Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, Devi Parikh, VQA: Visual Question Answering, CVPR, 2015 SUNw:Scene Understanding workshop.
* MPI / Berkeley [[Web]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/visual-turing-challenge/) [[Paper]](http://arxiv.org/pdf/1505.01121)
  * Mateusz Malinowski, Marcus Rohrbach, Mario Fritz, Ask Your Neurons: A Neural-based Approach to Answering Questions about Images, arXiv:1505.01121.
* Toronto [[Paper]](http://arxiv.org/pdf/1505.02074) [[Dataset]](http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/)
  * Mengye Ren, Ryan Kiros, Richard Zemel, Image Question Answering: A Visual Semantic Embedding Model and a New Dataset, arXiv:1505.02074 / ICML 2015 deep learning workshop.
* Baidu / UCLA [[Paper]](http://arxiv.org/pdf/1505.05612) [[Dataset]]()
  * Hauyuan Gao, Junhua Mao, Jie Zhou, Zhiheng Huang, Lei Wang, Wei Xu, Are You Talking to a Machine? Dataset and Methods for Multilingual Image Question Answering, arXiv:1505.05612.
* POSTECH [[Paper](http://arxiv.org/pdf/1511.05756.pdf)] [[Project Page](http://cvlab.postech.ac.kr/research/dppnet/)]
  * Hyeonwoo Noh, Paul Hongsuck Seo, and Bohyung Han, Image Question Answering using Convolutional Neural Network with Dynamic Parameter Prediction, arXiv:1511.05765
* CMU / Microsoft Research [[Paper](http://arxiv.org/pdf/1511.02274v2.pdf)]
  * Yang, Z., He, X., Gao, J., Deng, L., & Smola, A. (2015). Stacked Attention Networks for Image Question Answering. arXiv:1511.02274.
* MetaMind [[Paper](http://arxiv.org/pdf/1603.01417v1.pdf)]
  * Xiong, Caiming, Stephen Merity, and Richard Socher. "Dynamic Memory Networks for Visual and Textual Question Answering." arXiv:1603.01417 (2016).
* SNU + NAVER [[Paper](http://arxiv.org/abs/1606.01455)]
  * Jin-Hwa Kim, Sang-Woo Lee, Dong-Hyun Kwak, Min-Oh Heo, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang, *Multimodal Residual Learning for Visual QA*, arXiv:1606:01455
* UC Berkeley + Sony [[Paper](https://arxiv.org/pdf/1606.01847)]
  * Akira Fukui, Dong Huk Park, Daylen Yang, Anna Rohrbach, Trevor Darrell, and Marcus Rohrbach, *Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding*, arXiv:1606.01847
* Postech [[Paper](http://arxiv.org/pdf/1606.03647.pdf)]
  * Hyeonwoo Noh and Bohyung Han, *Training Recurrent Answering Units with Joint Loss Minimization for VQA*, arXiv:1606.03647
* SNU + NAVER [[Paper](http://arxiv.org/abs/1610.04325)]
  * Jin-Hwa Kim, Kyoung Woon On, Jeonghee Kim, Jung-Woo Ha, Byoung-Tak Zhang, *Hadamard Product for Low-rank Bilinear Pooling*, arXiv:1610.04325.

### Image Generation
* Convolutional / Recurrent Networks
  * Aäron van den Oord, Nal Kalchbrenner, Oriol Vinyals, Lasse Espeholt, Alex Graves, Koray Kavukcuoglu. "Conditional Image Generation with PixelCNN Decoders"[[Paper]](https://arxiv.org/pdf/1606.05328v2.pdf)[[Code]](https://github.com/kundan2510/pixelCNN)
  * Alexey Dosovitskiy, Jost Tobias Springenberg, Thomas Brox, "Learning to Generate Chairs with Convolutional Neural Networks", CVPR, 2015. [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf)
  * Karol Gregor, Ivo Danihelka, Alex Graves, Danilo Jimenez Rezende, Daan Wierstra, "DRAW: A Recurrent Neural Network For Image Generation", ICML, 2015. [[Paper](https://arxiv.org/pdf/1502.04623v2.pdf)] 
* Adversarial Networks
  * Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio, Generative Adversarial Networks, NIPS, 2014. [[Paper]](http://arxiv.org/abs/1406.2661)
  * Emily Denton, Soumith Chintala, Arthur Szlam, Rob Fergus, Deep Generative Image Models using a ￼Laplacian Pyramid of Adversarial Networks, NIPS, 2015. [[Paper]](http://arxiv.org/abs/1506.05751)
  * Lucas Theis, Aäron van den Oord, Matthias Bethge, "A note on the evaluation of generative models", ICLR 2016. [[Paper](http://arxiv.org/abs/1511.01844)]
  * Zhenwen Dai, Andreas Damianou, Javier Gonzalez, Neil Lawrence, "Variationally Auto-Encoded Deep Gaussian Processes", ICLR 2016. [[Paper](http://arxiv.org/pdf/1511.06455v2.pdf)]
  * Elman Mansimov, Emilio Parisotto, Jimmy Ba, Ruslan Salakhutdinov, "Generating Images from Captions with Attention", ICLR 2016, [[Paper](http://arxiv.org/pdf/1511.02793v2.pdf)]
  * Jost Tobias Springenberg, "Unsupervised and Semi-supervised Learning with Categorical Generative Adversarial Networks", ICLR 2016, [[Paper](http://arxiv.org/pdf/1511.06390v1.pdf)]
  * Harrison Edwards, Amos Storkey, "Censoring Representations with an Adversary", ICLR 2016, [[Paper](http://arxiv.org/pdf/1511.05897v3.pdf)]
  * Takeru Miyato, Shin-ichi Maeda, Masanori Koyama, Ken Nakae, Shin Ishii, "Distributional Smoothing with Virtual Adversarial Training", ICLR 2016, [[Paper](http://arxiv.org/pdf/1507.00677v8.pdf)]
  * Jun-Yan Zhu, Philipp Krahenbuhl, Eli Shechtman, and Alexei A. Efros, "Generative Visual Manipulation on the Natural Image Manifold", ECCV 2016. [[Paper](https://arxiv.org/pdf/1609.03552v2.pdf)] [[Code](https://github.com/junyanz/iGAN)] [[Video](https://youtu.be/9c4z6YsBGQ0)]
* Mixing Convolutional and Adversarial Networks
  * Alec Radford, Luke Metz, Soumith Chintala, "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks", ICLR 2016. [[Paper](http://arxiv.org/pdf/1511.06434.pdf)]

### Other Topics
* Visual Analogy [[Paper](https://web.eecs.umich.edu/~honglak/nips2015-analogy.pdf)]
  * Scott Reed, Yi Zhang, Yuting Zhang, Honglak Lee, Deep Visual Analogy Making, NIPS, 2015
* Surface Normal Estimation [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Designing_Deep_Networks_2015_CVPR_paper.pdf)
  * Xiaolong Wang, David F. Fouhey, Abhinav Gupta, Designing Deep Networks for Surface Normal Estimation, CVPR, 2015.
* Action Detection [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Gkioxari_Finding_Action_Tubes_2015_CVPR_paper.pdf)
  * Georgia Gkioxari, Jitendra Malik, Finding Action Tubes, CVPR, 2015.
* Crowd Counting [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Cross-Scene_Crowd_Counting_2015_CVPR_paper.pdf)
  * Cong Zhang, Hongsheng Li, Xiaogang Wang, Xiaokang Yang, Cross-scene Crowd Counting via Deep Convolutional Neural Networks, CVPR, 2015.
* 3D Shape Retrieval [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wang_Sketch-Based_3D_Shape_2015_CVPR_paper.pdf)
  * Fang Wang, Le Kang, Yi Li, Sketch-based 3D Shape Retrieval using Convolutional Neural Networks, CVPR, 2015.
* Weakly-supervised Classification
  * Samaneh Azadi, Jiashi Feng, Stefanie Jegelka, Trevor Darrell, "Auxiliary Image Regularization for Deep CNNs with Noisy Labels", ICLR 2016, [[Paper](http://arxiv.org/pdf/1511.07069v2.pdf)]
* Artistic Style [[Paper]](http://arxiv.org/abs/1508.06576) [[Code]](https://github.com/jcjohnson/neural-style)
  * Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, A Neural Algorithm of Artistic Style.
* Human Gaze Estimation
  * Xucong Zhang, Yusuke Sugano, Mario Fritz, Andreas Bulling, Appearance-Based Gaze Estimation in the Wild, CVPR, 2015. [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhang_Appearance-Based_Gaze_Estimation_2015_CVPR_paper.pdf) [[Website]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild-mpiigaze/)
* Face Recognition
  * Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf, DeepFace: Closing the Gap to Human-Level Performance in Face Verification, CVPR, 2014. [[Paper]](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)
  * Yi Sun, Ding Liang, Xiaogang Wang, Xiaoou Tang, DeepID3: Face Recognition with Very Deep Neural Networks, 2015. [[Paper]](http://arxiv.org/abs/1502.00873)
  * Florian Schroff, Dmitry Kalenichenko, James Philbin, FaceNet: A Unified Embedding for Face Recognition and Clustering, CVPR, 2015. [[Paper]](http://arxiv.org/abs/1503.03832)
* Facial Landmark Detection
  * Yue Wu, Tal Hassner, KangGeon Kim, Gerard Medioni, Prem Natarajan, Facial Landmark Detection with Tweaked Convolutional Neural Networks, 2015. [[Paper]](http://arxiv.org/abs/1511.04031) [[Project]](http://www.openu.ac.il/home/hassner/projects/tcnn_landmarks/)

## Courses
* Deep Vision
  * [Stanford] [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
  * [CUHK] [ELEG 5040: Advanced Topics in Signal Processing(Introduction to Deep Learning)](https://piazza.com/cuhk.edu.hk/spring2015/eleg5040/home)
* More Deep Learning
  * [Stanford] [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)
  * [Oxford] [Deep Learning by Prof. Nando de Freitas](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)
  * [NYU] [Deep Learning by Prof. Yann LeCun](http://cilvr.cs.nyu.edu/doku.php?id=courses:deeplearning2014:start)

## Books
* Free Online Books
  * [Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](http://www.iro.umontreal.ca/~bengioy/dlbook/)
  * [Neural Networks and Deep Learning by Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
  * [Deep Learning Tutorial by LISA lab, University of Montreal](http://deeplearning.net/tutorial/deeplearning.pdf)

## Videos
* Talks
  * [Deep Learning, Self-Taught Learning and Unsupervised Feature Learning By Andrew Ng](https://www.youtube.com/watch?v=n1ViNeWhC24)
  * [Recent Developments in Deep Learning By Geoff Hinton](https://www.youtube.com/watch?v=vShMxxqtDDs)
  * [The Unreasonable Effectiveness of Deep Learning by Yann LeCun](https://www.youtube.com/watch?v=sc-KbuZqGkI)
  * [Deep Learning of Representations by Yoshua bengio](https://www.youtube.com/watch?v=4xsVFLnHC_0)


## Software
### Framework
* Tensorflow: An open source software library for numerical computation using data flow graph by Google [[Web](https://www.tensorflow.org/)]
* Torch7: Deep learning library in Lua, used by Facebook and Google Deepmind [[Web](http://torch.ch/)]
  * Torch-based deep learning libraries: [[torchnet](https://github.com/torchnet/torchnet)],
* Caffe: Deep learning framework by the BVLC [[Web](http://caffe.berkeleyvision.org/)]
* Theano: Mathematical library in Python, maintained by LISA lab [[Web](http://deeplearning.net/software/theano/)]
  * Theano-based deep learning libraries: [[Pylearn2](http://deeplearning.net/software/pylearn2/)], [[Blocks](https://github.com/mila-udem/blocks)], [[Keras](http://keras.io/)], [[Lasagne](https://github.com/Lasagne/Lasagne)]
* MatConvNet: CNNs for MATLAB [[Web](http://www.vlfeat.org/matconvnet/)]
* MXNet: A flexible and efficient deep learning library for heterogeneous distributed systems with multi-language support [[Web](http://mxnet.io/)]
* Deepgaze: A computer vision library for human-computer interaction based on CNNs [[Web](https://github.com/mpatacchiola/deepgaze)]

### Applications
* Adversarial Training
  * Code and hyperparameters for the paper "Generative Adversarial Networks" [[Web]](https://github.com/goodfeli/adversarial)
* Understanding and Visualizing
  * Source code for "Understanding Deep Image Representations by Inverting Them," CVPR, 2015. [[Web]](https://github.com/aravindhm/deep-goggle)
* Semantic Segmentation
  * Source code for the paper "Rich feature hierarchies for accurate object detection and semantic segmentation," CVPR, 2014. [[Web]](https://github.com/rbgirshick/rcnn)
  * Source code for the paper "Fully Convolutional Networks for Semantic Segmentation," CVPR, 2015. [[Web]](https://github.com/longjon/caffe/tree/future)
* Super-Resolution
  * Image Super-Resolution for Anime-Style-Art [[Web]](https://github.com/nagadomi/waifu2x)
* Edge Detection
  * Source code for the paper "DeepContour: A Deep Convolutional Feature Learned by Positive-Sharing Loss for Contour Detection," CVPR, 2015. [[Web]](https://github.com/shenwei1231/DeepContour)
  * Source code for the paper "Holistically-Nested Edge Detection", ICCV 2015. [[Web]](https://github.com/s9xie/hed)

## Tutorials
* [CVPR 2014] [Tutorial on Deep Learning in Computer Vision](https://sites.google.com/site/deeplearningcvpr2014/)
* [CVPR 2015] [Applied Deep Learning for Computer Vision with Torch](https://github.com/soumith/cvpr2015)

## Blogs
* [Deep down the rabbit hole: CVPR 2015 and beyond@Tombone's Computer Vision Blog](http://www.computervisionblog.com/2015/06/deep-down-rabbit-hole-cvpr-2015-and.html)
* [CVPR recap and where we're going@Zoya Bylinskii (MIT PhD Student)'s Blog](http://zoyathinks.blogspot.kr/2015/06/cvpr-recap-and-where-were-going.html)
* [Facebook's AI Painting@Wired](http://www.wired.com/2015/06/facebook-googles-fake-brains-spawn-new-visual-reality/)
* [Inceptionism: Going Deeper into Neural Networks@Google Research](http://googleresearch.blogspot.kr/2015/06/inceptionism-going-deeper-into-neural.html)
* [Implementing Neural networks](http://peterroelants.github.io/) 
