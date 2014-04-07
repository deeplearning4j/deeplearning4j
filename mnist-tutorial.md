---
title: 
layout: default
---

*previous* - [mnist for restricted Boltzmann machines](../rbm-mnist.html)
# mnist for deep-belief networks

MNIST is a good place to begin exploring image recognition. 

# tutorial

To begin with, you’ll take an image from your data set and binarize it, which means you’ll convert its pixels from continuous gray scale to ones and zeros. A useful rule of thumb if that every gray-scale pixel with a value higher than 35 becomes a 1, and the rest are set to 0. The tool you’ll use to do that is an MNIST data-set iterator class.

The [MnistDataSetIterator](../doc/org/datasets/iterator/impl/MnistDataSetIterator.html) does this for you.

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


Note to windows uers, in place of the line below, please do the following:


         1. Download the preserialized mnist dataset [https://drive.google.com/file/d/0B-O_wola53IsWDhCSEtJWXUwTjg/edit?usp=sharing](here):

         2. Use the following dataset iterator, this one is equivalent to the one below:    

               DataSet d = new DataSet();
               BufferedInputStream bis = new BufferedInputStream(new FileInputStream(new File("path/to/your/file")));
               d.load(bis);
               bis.close();

          DataSetIterator iter = new ListDataSetIterator(d.asList(),10);

Next, we want to train a deep-belief network to reconstruct the MNIST data set. This is done with following snippet:

         //Train on batches of 10 out of 60000
         //Unix only
         DataSetIterator mnistData = new MnistDataSetIterator(10,60000);
        
	        //obtain the number of columns directly, this allows you to be agnostic to the number of training input columns.
	        DataSet first = fetcher.next();
			int numIns = first.getFirst().columns;
			int numLabels = first.getSecond().columns;
			//you may want to tune the number of layers here
			int[] layerSizes = {500,500,500};
			double lr = 0.001;
	        /*
	         *  Build the dbn with the number of inputs, don't render weights (otherwise specify the number of training epochs you want to display weights on)
	         *  No momentun, regularization, numLabels is 10 (0-9)
	         *
	         */
			DBN dbn = new DBN.Builder().numberOfInputs(numIns)
					.renderWeights(0).withMomentum(0).useRegularization(false)
					.numberOfOutPuts(numLabels).withRng(new MersenneTwister(123))
					.hiddenLayerSizes(layerSizes).build();
	      
	        //pretrain and finetune the network on the first input, then the next ones where necessary
			do  {
				dbn.pretrain(first.getFirst(),1, lr, 1000);
	             dbn.finetune(first.getSecond(),lr, 1000);
				
				if(fetcher.hasNext())
					first = fetcher.next();
			} while(fetcher.hasNext());

			
	       //reset the iterator; we will now calculate f-scores
	       //note that this is for demo purposes only, you would typically do a test set and cross validation
			fetcher.reset();
			first = fetcher.next();
			Evaluation eval = new Evaluation();

	        //test on the data set
			do {

					DoubleMatrix predicted = dbn.predict(first.getFirst());
					log.info("Predicting\n " + first.getSecond().toString().replaceAll(";","\n"));
					log.info("Prediction was " + predicted.toString().replaceAll(";","\n"));
					eval.eval(first.getSecond(), predicted);
					if(fetcher.hasNext())
						first = fetcher.next();
				}while(fetcher.hasNext());
		
			System.out.println(eval.stats());

Now that you've seen Deeplearning4j train a neural network on MNIST images, you may want to learn how to feed it other [datasets](../data-sets-ml.html).
