---
title: 
layout: default
---

* [DeepFace: Closing the Gap to Human-Level Performance in Face Verification](https://research.facebook.com/publications/480567225376225/deepface-closing-the-gap-to-human-level-performance-in-face-verification/); Yaniv TaigmanMing YangMarc'Aurelio RanzatoLior Wolf
Conference on Computer Vision and Pattern Recognition (CVPR); June 24, 2014
--Reaches an accuracy of 97.35% on the Labeled Faces in the Wild (LFW) dataset, reducing the error of the current state of the art by more than 27%, closely approaching human-level performance (~97.53 percent).

* When Google adopted deep-learning-based speech recognition in its Android smartphone operating system, it achieved a 25% reduction in word errors.

Kaggle President @JeremyPHoward: Recently, most winners used either Ensembles of decision trees (random forests) or Deep Learning #kdd2013
http://www.kdnuggets.com/2013/08/top-tweets-aug12-13.html

In 2012, the pharmaceutical company Merck offered a prize to whoever could beat its best programs for helping to predict useful drug candidates. The task was to trawl through database entries on more than 30,000 small molecules, each of which had thousands of numerical chemical-property descriptors, and to try to predict how each one acted on 15 different target molecules. Dahl and his colleagues won $22,000 with a deep-learning system. “We improved on Merck's baseline by about 15%,” he says.

Malik remembers that Hinton asked him: “You're a sceptic. What would convince you?” Malik replied that a victory in the internationally renowned ImageNet competition might do the trick.
In that competition, teams train computer programs on a data set of about 1 million images that have each been manually labelled with a category. After training, the programs are tested by getting them to suggest labels for similar images that they have never seen before. They are given five guesses for each test image; if the right answer is not one of those five, the test counts as an error. Past winners had typically erred about 25% of the time. In 2012, Hinton's lab entered the first ever competitor to use deep learning. It had an error rate of just 15% (ref. 4).

Many others have followed: in 2013, all entrants to the ImageNet competition used deep learning.
“Over the next few years we'll see a feeding frenzy. Lots of people will jump on the deep-learning bandwagon.”


In the past three years, in addition to the IJCNN 2011 Traffic Sign Recognition Competition mentioned above, they won seven other highly competitive international visual pattern recognition contests:

ICPR 2012 Contest on “Mitosis Detection in Breast Cancer Histological Images.” This is important for breast cancer prognosis. Humans tend to find it very difficult to distinguish mitosis from other tissue. 129 companies, research institutes, and universities in 40 countries registered; 14 sent their results. Our NN won by a comfortable margin.

ISBI 2012 challenge on segmentation of neuronal structures. Given electron microscopy images of stacks of thin slices of animal brains, the goal is to build a detailed 3D model of the brain’s neurons and dendrites. But human experts need many hours to annotate the images: Which parts depict neuronal membranes? Which parts are irrelevant background? Our NNs learn to solve this task through experience with millions of training images. In March 2012, they won the contest on all three evaluation metrics by a large margin, with superhuman performance in terms of pixel error. (Ranks 2–6: for researchers at ETHZ, MIT, CMU, Harvard.) A NIPS 2012 paper on this is coming up.

ICDAR 2011Offline Chinese Handwriting Competition. Our team won the competition although none of its members speaks a word of Chinese. In the not-so-distant future you should be able to point your cell phone camera to text in a foreign language, and get a translation. That’s why we also developed low-power implementations of our NNs for cell phone chips.

Online German Traffic Sign Recognition Contest (2011, first and second rank). Until the last day of the competition, we thought we had a comfortable lead, but then our toughest competitor from NYU surged ahead, and our team (with Dan Ciresan, Ueli Meier, Jonathan Masci) had to work late-night to re-establish the correct order. :)
ICDAR 2009 Arabic Connected Handwriting Competition (although none of us speaks a word of Arabic).
ICDAR 2009 Handwritten Farsi/Arabic Character Recognition Competition (idem).
ICDAR 2009 French Connected Handwriting Competition. Our French also isn’t that good. :)


A few familiar names win the Kaggle Merck Molecular Activity Challenge: George Dahl, +Ruslan Salakhutdinov , +Navdeep Jaitly, Christopher Jordan-Squire and +Geoffrey Hinton, using deep learning and the 'dropout' approach. A nice summary by George Dahl here:
http://blog.kaggle.com/2012/10/31/merck-competition-results-deep-nn-and-gpus-come-out-to-play/
Team gggg, made up of 5 Kaggle newcomers, dominated the final two weeks of the competition by using deep learning algorithms running on GPUs, both Kaggle firsts. Led by George Dahl, a doctoral student at the University of Toronto, the team used the competition to illustrate the ability of neural network models to perform well with no feature engineering and only minimal preprocessing. After his previous experiences applying deep learning techniques to speech recognition and language processing tasks, George was drawn to the complexity of the Merck data set and the challenge of working in a new data domain. He assembled a team of heavy hitters from the world of machine learning and neural networks. Ruslan Salakhutdinov, an assistant professor in statistics and computer science at Toronto, specializes in Bayesian statistics, probabilistic graphical models, and large-scale optimization. Navdeep Jaitly, a doctoral student at the University of Toronto who works on applying deep learning to problems in speech recognition, took interest due to his background in computational biology and proteomics. Christopher Jordan-Squire, a doctoral student in mathematics at the University of Washington, studies constrained optimization applied to statistics and machine learning and joined to get a break from proving theorems. Finally, they were advised by Professor Geoffrey Hinton, perhaps best known as one of the inventors of the back-propagation algorithm. Geoff, the Ph.D. advisor to George and Navdeep, joined the team to help demonstrate the power of deep neural networks that use dropout, although his direct contribution to this competition was limited to making suggestions to George. 

Yann LeCun, a leading researcher on Deep Learning, who was recently hired by Facebook to head their AI Lab,  reports that his  former student +Pierre Sermanet won the Dogs vs Cats competition on Kaggle. 
Pierre entry was amazingly good - 98.9% correct. He posted on Google+
I just won the Dogs vs. Cats Kaggle competition, using the deep learning library I wrote during my PhD: OverFeat
