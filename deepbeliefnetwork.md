---
title: 
layout: default
---

# deep-belief network

For our purposes, a [deep-belief network](http://www.scholarpedia.org/article/Deep_belief_networks) can be defined as a stack of restricted Boltzmann machines in which each layer communicates with its previous and subsequent layers, while the nodes of any single layer do not communicate with each other laterally. With the exception of the  first and final layers, each hidden layer has a double role: it serves as the hidden layer to the higher nodes before, and as the input layer to the lower nodes after. It is a network of networks. 

Deep-belief networks are used to recognize and generate images, video sequences and motion-capture data. 

### parameters

Please also see [the multilayer network parameters common to all multilayer networks]({{ site.baseurl }}/multinetwork.html)

#### k 

K is the number of times you run [contrastive divergence]({{ site.baseurl }}/glossary.html#contrastivedivergence). Each time contrastive divergence is run, it is a sample of the Markov chain. In composing the restricted Boltzmann machine, a typical value is one.

### initiating a deep-belief network

Setting up a single-thread deep belief network is easy. 

To create the machine, you simply instantiate an object of the class [DBN]({{ site.baseurl }}/doc/com/ccc/deeplearning/dbn/DBN.html).


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

It creates a deep-belief network with the specified hidden layer sizes (three hidden layers at two hidden units each); the number of inputs being two; two outputs; no regularization; the specified random number generator; and no momentum.

Next, your create a training set for the machine. For the sake of visual brevity, a toy, two-dimensional data set is included in the code below. (With large-scale projects, training sets are clearly much more substantial.)

        int n = 10;
		DataSet d = MatrixUtil.xorData(n);
		DoubleMatrix x = d.getFirst();
		DoubleMatrix y = d.getSecond();

An xor dataset is generated here with 10 columns. A data set is a pair of x,y matrices such that each matrix is one row.

        dbn.pretrain(x,k, preTrainLr, preTrainEpochs);
		dbn.finetune(y,fineTuneLr, fineTuneEpochs);

Those pretraining and finetuning steps train the network. You can test your trained network by feeding it unstructured data and checking the output. The output here will be a prediction of whether the specified input is true or false based on the rules of xor.


		DoubleMatrix predict = dbn.predict(x);

		Evaluation eval = new Evaluation();
		eval.eval(y, predict);
		System.out.println(eval.stats());


This will print out the f score of the prediction.

Note that the eval class combines confusion matrices and f1 scores to allow for easy display and evaluation of data by allowing input of outcome matrices. This is useful for tracking how well your network trains over time. The f1 score will be a percentage. It's basically the probability that your guess are correct correct. Eighty-six percent or more is pretty good.

If you run into trouble, try modifying the hidden layer sizes, and tweaking other parameters to get your f1 score up.

