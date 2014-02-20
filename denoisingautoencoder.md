---
title: 
layout: default
---


# denoising autoencoder

An autoencoder is a neural network used for dimensionality reduction; that is, for feature selection and extraction. Autoencoders with more hidden layers than inputs run the risk of learning the [identity function](https://en.wikipedia.org/wiki/Identity_function) -- where the output simple equals the input -- thereby becoming useless. 

Denoising autoencoders are an extension of the basic autoencoder, and represent a stochastic version of the autoencoder. Denoising autoencoders attempt to address identity-function risk by randomly corrupting input (i.e. introducing noise) that the autoencoder must then reconstruct, or denoise. 

### parameters

#### corruption level 

The amount of noise to apply to the input takes the form of a percentage. Typically, 30 percent (0.3) is fine, but if you have very little data, you may want to consider adding more.

## input

### initiating a denoising autoencoder

Setting up a single-thread denoising autoencoder is easy. 

To create the machine, you simply instantiate an object of the [class]({{ site.baseurl }}/doc/com/ccc/deeplearning/da/DenoisingAutoEncoder.html).



    			DenoisingAutoEncoder da = new DenoisingAutoEncoder.Builder().numberOfVisible(1).numHidden(1).build();



 This sets up denoising autoencoder with 1 visible layer and 1 hidden layer.







Next, create a training set for the machine. For the sake of visual brevity, a toy, two-dimensional data set is included in the code below. (With large-scale projects, training sets are clearly much more substantial.)

		DataSet xor = MatrixUtil.xorData(10);
		DenoisingAutoEncoder da = new DenoisingAutoEncoder.Builder().numberOfVisible(xor.getFirst().columns).numHidden(2).build();
        
        double lr = 0.1;
	   da.trainTillConvergence(xor.getFirst(), lr, 0.3); 

This will train the specified denoising autoencoder with a corruption level of 0.3 and a learning rate of 0.1 learning rate on the x matrix in the dataset.




You can test your input with the following:

    System.out.println(da.reconstruct(xor.getFirst()));


 If you see percentages where 1s are in the original input that is a good indicator your autoencoder

 is learning the structure of the data.


