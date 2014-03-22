---
title: 
layout: default
---

# deep-belief networks

For our purposes, a [deep-belief network](http://www.scholarpedia.org/article/Deep_belief_networks) can be defined as a stack of restricted Boltzmann machines in which each layer communicates with both its previous and subsequent layers. The nodes of any single layer do not communicate with each other laterally. 

With the exception of the first and final layers, each hidden layer has a double role: it serves as the hidden layer to the higher nodes that come before, and as the input layer to the lower nodes after. It is a network of single-layer networks. 

Deep-belief networks are used to recognize and generate images, video sequences and motion-capture data. 

### parameters & k

See the [parameters common to all multilayer networks](../multinetwork.html).

The variable k is the number of times you run [contrastive divergence](../glossary.html#contrastivedivergence). Each time contrastive divergence is run, it's a sample of the Markov chain. In composing a deep-belief network, a typical value is one.

### initiating a deep-belief network

Here's how you set up a single-thread deep belief network: 

To create it, you instantiate an object of the class [DBN](../doc/org/deeplearning4j/dbn/DBN.html).

	        RandomGenerator rng = new MersenneTwister(123);

			double preTrainLr = 0.001;
			int preTrainEpochs = 1000;
			int k = 1;
			int nIns = 2,nOuts = 2;
			int[] hiddenLayerSizes = new int[] {2,2,2};
			double fineTuneLr = 0.001;
			int fineTuneEpochs = 1000;

	        DBN dbn = new DBN.Builder().hiddenLayerSizes(hiddenLayerSizes).numberOfInputs(nIns)
			.useRegularization(false).withMomentum(0)
			.numberOfOutPuts(nOuts).withRng(rng).build();


This is a little more complicated than the singular input. 

It creates a deep-belief network with the specified hidden-layer sizes (three hidden layers at two hidden units each); the number of inputs being two; outputs also two; no regularization; the specified random number generator; and no momentum.

Next, you create a training set for the machine. For the sake of visual brevity, a toy, two-dimensional data set is included in the code below. (With large-scale projects, training sets are clearly much larger.)

	        int n = 10;
			DataSet d = MatrixUtil.xorData(n);
			DoubleMatrix x = d.getFirst();
			DoubleMatrix y = d.getSecond();

An xor dataset is generated here with 10 columns. A data set is a pair of x,y matrices such that each matrix is one row.

	        dbn.pretrain(x,k, preTrainLr, preTrainEpochs);
			dbn.finetune(y,fineTuneLr, fineTuneEpochs);

Pretraining and finetuning steps train the network for use on unstructured data. You can test the trained network by feeding it unstructured data and checking the output. The output here will be a prediction of whether the specified input is true or false based on the rules of xor.

			DoubleMatrix predict = dbn.predict(x);

			Evaluation eval = new Evaluation();
			eval.eval(y, predict);
			System.out.println(eval.stats());

This will print out the f1 score of the prediction.

Note that the eval class combines [confusion matrices](../glossary.html#confusionmatrix) and f1 scores to allow for easy display and evaluation of data by allowing input of outcome matrices. This is useful for tracking how well your network trains over time. 

The f1 score will be a percentage. It's basically the probability that your guess are correct correct. Eighty-six percent is industry standard; a solid deep-learning network should be capable of scores in the high 90s.

If you run into trouble, try modifying the hidden layer sizes, and tweaking other parameters to get the f1 score up.

Click [here](../continuousdeepbeliefnetwork.html) to learn how to set up continuous deep-belief networks. (CDBNs accept a continuum of integers rather than binary data.)

Next, we'll show you how to use [distributed and multithreaded computing](../scaleout.html) to train your networks more quickly.