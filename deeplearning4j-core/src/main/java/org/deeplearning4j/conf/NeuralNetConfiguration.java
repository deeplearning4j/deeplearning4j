package org.deeplearning4j.conf;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.HiddenLayer;
import org.deeplearning4j.nn.LogisticRegression;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.nn.activation.Sigmoid;
import org.deeplearning4j.optimize.MultiLayerNetworkOptimizer;
import org.deeplearning4j.transformation.MatrixTransform;
import org.jblas.DoubleMatrix;


public class NeuralNetConfiguration implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8267028988938122369L;

	//number of columns in the input matrix
	public int nIns;
	//the hidden layer sizes at each layer
	public int[] hiddenLayerSizes;
	//the number of outputs/labels for logistic regression
	public int nOuts;
	public int nLayers;
	//logistic regression output layer (aka the softmax layer) for translating network outputs in to probabilities
	public LogisticRegression logLayer;
	public RandomGenerator rng;
	/* probability distribution for generation of weights */
	public RealDistribution dist;
	public double momentum = 0.1;
	public MultiLayerNetworkOptimizer optimizer;
	public ActivationFunction activation = new Sigmoid();
	public boolean toDecode;
	public double l2 = 0.01;
	public boolean shouldInit = true;
	public double fanIn = -1;
	public int renderWeightsEveryNEpochs = -1;
	public boolean useRegularization = true;
	protected Map<Integer,MatrixTransform> weightTransforms = new HashMap<Integer,MatrixTransform>();
	protected boolean shouldBackProp = true;
	protected boolean forceNumEpochs = false;
	public int getnIns() {
		return nIns;
	}
	public void setnIns(int nIns) {
		this.nIns = nIns;
	}
	public int[] getHiddenLayerSizes() {
		return hiddenLayerSizes;
	}
	public void setHiddenLayerSizes(int[] hiddenLayerSizes) {
		this.hiddenLayerSizes = hiddenLayerSizes;
	}
	public int getnOuts() {
		return nOuts;
	}
	public void setnOuts(int nOuts) {
		this.nOuts = nOuts;
	}
	public int getnLayers() {
		return nLayers;
	}
	public void setnLayers(int nLayers) {
		this.nLayers = nLayers;
	}
	public LogisticRegression getLogLayer() {
		return logLayer;
	}
	public void setLogLayer(LogisticRegression logLayer) {
		this.logLayer = logLayer;
	}
	public RandomGenerator getRng() {
		return rng;
	}
	public void setRng(RandomGenerator rng) {
		this.rng = rng;
	}
	public RealDistribution getDist() {
		return dist;
	}
	public void setDist(RealDistribution dist) {
		this.dist = dist;
	}
	public double getMomentum() {
		return momentum;
	}
	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}
	public MultiLayerNetworkOptimizer getOptimizer() {
		return optimizer;
	}
	public void setOptimizer(MultiLayerNetworkOptimizer optimizer) {
		this.optimizer = optimizer;
	}
	public ActivationFunction getActivation() {
		return activation;
	}
	public void setActivation(ActivationFunction activation) {
		this.activation = activation;
	}
	public boolean isToDecode() {
		return toDecode;
	}
	public void setToDecode(boolean toDecode) {
		this.toDecode = toDecode;
	}
	public double getL2() {
		return l2;
	}
	public void setL2(double l2) {
		this.l2 = l2;
	}
	public boolean isShouldInit() {
		return shouldInit;
	}
	public void setShouldInit(boolean shouldInit) {
		this.shouldInit = shouldInit;
	}
	public double getFanIn() {
		return fanIn;
	}
	public void setFanIn(double fanIn) {
		this.fanIn = fanIn;
	}
	public int getRenderWeightsEveryNEpochs() {
		return renderWeightsEveryNEpochs;
	}
	public void setRenderWeightsEveryNEpochs(int renderWeightsEveryNEpochs) {
		this.renderWeightsEveryNEpochs = renderWeightsEveryNEpochs;
	}
	public boolean isUseRegularization() {
		return useRegularization;
	}
	public void setUseRegularization(boolean useRegularization) {
		this.useRegularization = useRegularization;
	}
	public Map<Integer, MatrixTransform> getWeightTransforms() {
		return weightTransforms;
	}
	public void setWeightTransforms(Map<Integer, MatrixTransform> weightTransforms) {
		this.weightTransforms = weightTransforms;
	}
	public boolean isShouldBackProp() {
		return shouldBackProp;
	}
	public void setShouldBackProp(boolean shouldBackProp) {
		this.shouldBackProp = shouldBackProp;
	}
	public boolean isForceNumEpochs() {
		return forceNumEpochs;
	}
	public void setForceNumEpochs(boolean forceNumEpochs) {
		this.forceNumEpochs = forceNumEpochs;
	}
	
	
	
	

}
