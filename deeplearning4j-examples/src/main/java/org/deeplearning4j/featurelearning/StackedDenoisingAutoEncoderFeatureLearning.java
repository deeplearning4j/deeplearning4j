package org.deeplearning4j.featurelearning;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.distributions.Distributions;
import org.deeplearning4j.models.classifiers.sda.StackedDenoisingAutoEncoder;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class StackedDenoisingAutoEncoderFeatureLearning {
	
	public static void main(String[] args) throws Exception {
		 RandomGenerator gen = new MersenneTwister(123);
	        MnistDataFetcher fetcher = new MnistDataFetcher(true);
	        fetcher.fetch(30000);
	        DataSet d2 = fetcher.next();

	        NeuralNetConfiguration conf = new NeuralNetConfiguration.Builder()
	                .momentum(5e-1f).weightInit(WeightInit.DISTRIBUTION).dist(Distributions.uniform(gen, 784, 10))
	                .withActivationType(NeuralNetConfiguration.ActivationType.SAMPLE)
	                .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY).rng(gen)
	                .learningRate(1e-1f).nIn(784).nOut(784).build();

	        //note: denoising autoencoders are primarily meant for binary data. There is an extension
	        //by bengio for continuous data that needs to be implemented yet.
	        StackedDenoisingAutoEncoder d = new StackedDenoisingAutoEncoder.Builder().configure(conf)
	                .hiddenLayerSizes(new int[]{500, 250, 200,250,500})
	                .build();

	        d.getOutputLayer().conf().setActivationFunction(Activations.softMaxRows());
	        d.getOutputLayer().conf().setLossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY);
	        
	        
	        d2.setLabels(d2.getFeatureMatrix());
	        
	        d.fit(d2);
	        
	        

	}
	
	

}
