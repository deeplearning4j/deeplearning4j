package org.deeplearning4j.sda;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.da.DenoisingAutoEncoder;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.NeuralNetwork;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



/**
 * A JBlas implementation of 
 * stacked denoising auto encoders.
 * @author Adam Gibson
 *
 */
public class StackedDenoisingAutoEncoder extends BaseMultiLayerNetwork  {

	private static final long serialVersionUID = 1448581794985193009L;
	private static Logger log = LoggerFactory.getLogger(StackedDenoisingAutoEncoder.class);



	public StackedDenoisingAutoEncoder() {}

	public StackedDenoisingAutoEncoder(int n_ins, int[] hiddenLayerSizes, int nOuts,
			int nLayers, RandomGenerator rng, DoubleMatrix input,DoubleMatrix labels) {
		super(n_ins, hiddenLayerSizes, nOuts, nLayers, rng, input,labels);

	}


	public StackedDenoisingAutoEncoder(int nIns, int[] hiddenLayerSizes, int nOuts,
			int n_layers, RandomGenerator rng) {
		super(nIns, hiddenLayerSizes, nOuts, n_layers, rng);
	}


	public void pretrain( double lr,  double corruptionLevel,  int epochs) {
		pretrain(this.getInput(),lr,corruptionLevel,epochs);
	}

	
	@Override
	public void pretrain(DoubleMatrix input, Object[] otherParams) {
		if(otherParams == null) {
			otherParams = new Object[]{0.01,0.3,1000};
		}
		
		Double lr = (Double) otherParams[0];
		Double corruptionLevel = (Double) otherParams[1];
		Integer epochs = (Integer) otherParams[2];

		pretrain(input, lr, corruptionLevel, epochs);
		
	}
	
	/**
	 * Unsupervised pretraining based on reconstructing the input
	 * from a corrupted version
	 * @param input the input to train on
	 * @param lr the starting learning rate
	 * @param corruptionLevel the corruption level (the smaller number of inputs; the higher the 
	 * corruption level should be) the percent of inputs to corrupt
	 * @param epochs the number of iterations to run
	 */
	public void pretrain(DoubleMatrix input,double lr,  double corruptionLevel,  int epochs) {
		if(this.getInput() == null)
			initializeLayers(input.dup());

		DoubleMatrix layerInput = null;

		for(int i = 0; i < this.getnLayers(); i++) {  // layer-wise                        
			//input layer
			if(i == 0)
				layerInput = input;
			else
				layerInput = this.getSigmoidLayers()[i - 1].sampleHGivenV(layerInput);
			if(isForceNumEpochs()) {
				for(int epoch = 0; epoch < epochs; epoch++) {
					getLayers()[i].train(layerInput, lr,  new Object[]{corruptionLevel,lr});
					log.info("Error on epoch " + epoch + " for layer " + (i + 1) + " is " + getLayers()[i].getReConstructionCrossEntropy());
					getLayers()[i].epochDone(epoch);

				}
			}
			else
				getLayers()[i].trainTillConvergence(layerInput, lr, new Object[]{corruptionLevel,lr,epochs});


		}	
	}

	/**
	 * 
	 * @param input input examples
	 * @param labels output labels
	 * @param otherParams
	 * 
	 * (double) learningRate
	 * (double) corruptionLevel
	 * (int) epochs
	 * 
	 * Optional:
	 * (double) finetune lr
	 * (int) finetune epochs
	 * 
	 */
	@Override
	public void trainNetwork(DoubleMatrix input, DoubleMatrix labels,
			Object[] otherParams) {
		if(otherParams == null) {
			otherParams = new Object[]{0.01,0.3,1000};
		}
		
		Double lr = (Double) otherParams[0];
		Double corruptionLevel = (Double) otherParams[1];
		Integer epochs = (Integer) otherParams[2];

		pretrain(input, lr, corruptionLevel, epochs);
		if(otherParams.length <= 3)
			finetune(labels, lr, epochs);
		else {
			Double finetuneLr = (Double) otherParams[3];
			Integer fineTuneEpochs = (Integer) otherParams[4];
			finetune(labels,finetuneLr,fineTuneEpochs);
		}
	}



	@Override
	public NeuralNetwork createLayer(DoubleMatrix input, int nVisible,
			int nHidden, DoubleMatrix W, DoubleMatrix hbias,
			DoubleMatrix vBias, RandomGenerator rng,int index) {
		DenoisingAutoEncoder ret = new DenoisingAutoEncoder.Builder()
		.withHBias(hbias).withInput(input).withWeights(W).withDistribution(getDist())
		.withRandom(rng).withMomentum(getMomentum()).withVisibleBias(vBias)
		.numberOfVisible(nVisible).numHidden(nHidden).withDistribution(getDist())
		.withSparsity(this.getSparsity()).renderWeights(getRenderWeightsEveryNEpochs()).fanIn(getFanIn())
		.build();
		if(gradientListeners.get(index) != null)
			ret.setGradientListeners(gradientListeners.get(index));
		return ret;
	}


	@Override
	public NeuralNetwork[] createNetworkLayers(int numLayers) {
		return new DenoisingAutoEncoder[numLayers];
	}


	public static class Builder extends BaseMultiLayerNetwork.Builder<StackedDenoisingAutoEncoder> {
		public Builder() {
			this.clazz = StackedDenoisingAutoEncoder.class;
		}
	}


	





}