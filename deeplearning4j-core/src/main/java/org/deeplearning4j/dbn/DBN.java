package org.deeplearning4j.dbn;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.rbm.RBM;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Deep Belief Network. This is a MultiLayer Perceptron Model
 * using Restricted Boltzmann Machines.
 *  See Hinton's practical guide to RBMs for great examples on
 *  how to train and tune parameters.
 *  
 * @author Adam Gibson
 *
 */
public class DBN extends BaseMultiLayerNetwork {

	private static final long serialVersionUID = -9068772752220902983L;
	private static Logger log = LoggerFactory.getLogger(DBN.class);

	public DBN() {}


	public DBN(int nIns, int[] hiddenLayerSizes, int nOuts, int nLayers,
			RandomGenerator rng, DoubleMatrix input,DoubleMatrix labels) {
		super(nIns, hiddenLayerSizes, nOuts, nLayers, rng, input,labels);
	}



	public DBN(int n_ins, int[] hidden_layer_sizes, int nOuts, int nLayers,
			RandomGenerator rng) {
		super(n_ins, hidden_layer_sizes, nOuts, nLayers, rng);
	}


	/**
	 * 
	 * @param input input examples
	 * @param labels output labels
	 * @param otherParams
	 * 
	 * (int)    k
	 * (double) learningRate
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
		int k = (Integer) otherParams[0];
		double lr = (Double) otherParams[1];
		int epochs = (Integer) otherParams[2];
		pretrain(input,k,lr,epochs);
		if(otherParams.length < 3)
			finetune(labels, lr, epochs);
		else {
			double finetuneLr = otherParams.length > 3 ? (double) otherParams[3] : lr;
			int finetuneEpochs = otherParams.length > 4 ? (int) otherParams[4] : epochs;
			finetune(labels,finetuneLr,finetuneEpochs);
		}
	}





	@Override
	public void pretrain(DoubleMatrix input, Object[] otherParams) {
		int k = (Integer) otherParams[0];
		double lr = (Double) otherParams[1];
		int epochs = (Integer) otherParams[2];
		pretrain(input,k,lr,epochs);

	}


	/**
	 * This unsupervised learning method runs
	 * contrastive divergence on each RBM layer in the network.
	 * @param input the input to train on
	 * @param k the k to use for running the RBM contrastive divergence.
	 * The typical tip is that the higher k is the closer to the model
	 * you will be approximating due to more sampling. K = 1
	 * usually gives very good results and is the default in quite a few situations.
	 * @param learningRate the learning rate to use
	 * @param epochs the number of epochs to train
	 */
	public void pretrain(DoubleMatrix input,int k,double learningRate,int epochs) {

		if(this.getInput() == null || this.getLayers() == null || this.getLayers()[0] == null || this.getSigmoidLayers() == null || this.getSigmoidLayers()[0] == null) {
			setInput(input);
			initializeLayers(input);
		}
		else 
			setInput(input);

		DoubleMatrix layerInput = null;

		for(int i = 0; i < getnLayers(); i++) {
			if(i == 0)
				layerInput = this.getInput();
			else 
				layerInput = getSigmoidLayers()[i - 1].sampleHGivenV(layerInput);
			log.info("Training on layer " + (i + 1));
			if(isForceNumEpochs()) {
				for(int epoch = 0; epoch < epochs; epoch++) {
					log.info("Error on epoch " + epoch + " for layer " + (i + 1) + " is " + getLayers()[i].getReConstructionCrossEntropy());
					getLayers()[i].train(layerInput, learningRate,new Object[]{k,learningRate});
					getLayers()[i].epochDone(epoch);
				}
			}
			else
				getLayers()[i].trainTillConvergence(layerInput, learningRate, new Object[]{k,learningRate,epochs});


		}
	}

	public void pretrain(int k,double learningRate,int epochs) {
		pretrain(this.getInput(),k,learningRate,epochs);
	}


	@Override
	public NeuralNetwork createLayer(DoubleMatrix input, int nVisible,
			int nHidden, DoubleMatrix W, DoubleMatrix hBias,
			DoubleMatrix vBias, RandomGenerator rng,int index) {

		RBM ret = new RBM.Builder()
		.useRegularization(isUseRegularization())
		.useAdaGrad(isUseAdaGrad()).normalizeByInputRows(isNormalizeByInputRows())
		.withMomentum(getMomentum()).withSparsity(getSparsity()).withDistribution(getDist())
		.numberOfVisible(nVisible).numHidden(nHidden).withWeights(W)
		.withInput(input).withVisibleBias(vBias).withHBias(hBias).withDistribution(getDist())
		.withRandom(rng).renderWeights(getRenderWeightsEveryNEpochs())
		.fanIn(getFanIn()).build();
		
		ret.setGradientListeners(gradientListeners.get(index));
		return ret;
	}

	@Override
	public NeuralNetwork[] createNetworkLayers(int numLayers) {
		return new RBM[numLayers];
	}


	public static class Builder extends BaseMultiLayerNetwork.Builder<DBN> {
		public Builder() {
			this.clazz = DBN.class;
		}
	}




}
