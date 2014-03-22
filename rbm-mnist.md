---
title: 
layout: default
---

# mnist for restricted boltzmann machines

The MNIST database is a large set of handwritten digits used to train neural networks and other algorithms in image recognition. MNIST has 60,000 images in its training set and 10,000 in its test set. 

MNIST derives from NIST, and stands for “Mixed National Institute of Standards and Technology.” The MNIST database reshuffles the NIST database's thousands of binary images of handwritten digits in order to better train and test various image recognition techniques. A full explanation of why MNIST is preferable to NIST can be found on [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).

Each image in the MNIST database is a 28x28 pixel cell, and each cell is contained within a bounding box, the four lines of pixels that frame it. The image is centered according to the center of mass of its pixels. 

MNIST is a good place to begin exploring image recognition. Here’s an easy way to load the data and get started. 

### tutorial

To begin with, you’ll take an image from your data set and binarize it, which means you’ll convert its pixels from continuous gray scale to ones and zeros. A useful rule of thumb if that every gray-scale pixel with a value higher than 35 becomes a 1, and the rest are set to 0. The tool you’ll use to do that is an MNIST data-set iterator class.

The [MnistDataSetIterator](../docs/org/deeplearning4j/datasets/iterator/impl/MnistDataSetIterator.html) does this for you.

A DataSetIterator can be used like this:


         DataSetIterator iter = ....;

         while(iter.hasNext()) {
         	DataSet next = iter.next();
         	//do stuff with the data set
         }

Typically, a DataSetIterator handles inputs and data-set-specific concerns like binarizing or normalization. For MNIST, the following does the trick:
         
         //Train on batches of 10 out of 60000
         DataSetIterator mnistData = new MnistDataSetIterator(10,60000);

The reason we specify the batch size as well as the number of examples is so the user can choose how many examples they want to look at.

Next, we want to train a restricted Boltzmann machine to reconstruct the MNIST data set. This is done with following snippet:
        


	        //Create an RBM with non regularization, 784 inputs (number for mnist) and 500 outputs
			RBM rbm = new RBM.Builder().numberOfVisible(784).numHidden(500)
					.useRegularization(false).withMomentum(0).build();



	        //train over the data set
	        while(iter.hasNext()) {
	           DataSet first = fetcher.next();
	           rbm.trainTillConvergence(first.getFirst(), 0.01, new Object[]{1,0.01,1000});

	        }  

	        //Iterate over the data set after done training and show the 2 side by side (you have to drag the test image over to the right)
			for(int j = 0; j < first.numExamples(); j++) {
				DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
				DoubleMatrix reconstructed2 = reconstruct.getRow(j);
				DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);

				DrawMnistGreyScale d = new DrawMnistGreyScale(draw1);
				d.title = "REAL";
				d.draw();
				DrawMnistGreyScale d2 = new DrawMnistGreyScale(draw2,100,100);
				d2.title = "TEST";
				d2.draw();
				Thread.sleep(10000);
				d.frame.dispose();
				d2.frame.dispose();

			}

Next, we'll show you how to train a deep-belief network [to reconstruct and recognize the MNIST images](../mnist.html).