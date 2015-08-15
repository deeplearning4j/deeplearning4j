package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.GRUParamInitializer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

/** Gated Recurrent Unit RNN Layer.
 * @author Alex Black
 */
public class GRU extends BaseLayer {

	public GRU(NeuralNetConfiguration conf) {
		super(conf);
	}
	
	public GRU(NeuralNetConfiguration conf, INDArray input) {
		super(conf, input);
	}
	@Override
	public Gradient gradient() {
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public Gradient calcGradient(Gradient layerError, INDArray activation){
		throw new UnsupportedOperationException("Not yet implemented");
	}

	@Override
	public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
		
		throw new UnsupportedOperationException("Not yet implemented");
	}
	
	@Override
	public INDArray preOutput(INDArray x) {
		return activate(x,true);
	}

	@Override
	public INDArray preOutput(INDArray x, boolean training) {
		return activate(x, training);
	}

	@Override
	public INDArray activate(INDArray input, boolean training){
		setInput(input, training);
		return activateHelper(training)[0];
	}

	@Override
	public INDArray activate(INDArray input){
		setInput(input);
		return activateHelper(true)[0];
	}

	@Override
	public INDArray activate(boolean training){
		return activateHelper(training)[0];
	}

	@Override
	public INDArray activate(){
		return activateHelper()[0];
	}

	private INDArray[] activateHelper() {
		return activateHelper(false);
	}

	/** Returns activations array: {output,rucZs,rucAs} in that order. */
	private INDArray[] activateHelper(boolean training){
		
		INDArray inputWeights = getParam(GRUParamInitializer.INPUT_WEIGHTS); //Shape: [n^(L-1),3*n^L], order: [wr,wu,wc]
		INDArray recurrentWeights = getParam(GRUParamInitializer.RECURRENT_WEIGHTS);	//Shape: [n^L,3*n^L]; order: [wR,wU,wC]
		INDArray biases = getParam(GRUParamInitializer.BIAS); //Shape: [1,3*n^L]; order: [br,bu,bc]
		
		boolean is2dInput = input.rank() < 3;		//Edge case of T=1, may have shape [m,nIn], equiv. to [m,nIn,1]
		int timeSeriesLength = (is2dInput ? 1 : input.size(2));
		int hiddenLayerSize = recurrentWeights.size(0);
		int miniBatchSize = input.size(0);
		
		//Apply dropconnect to input (not recurrent) weights only:
		if(conf.isUseDropConnect() && training) {
			if (conf.getDropOut() > 0) {
				inputWeights = Dropout.applyDropConnect(this,GRUParamInitializer.INPUT_WEIGHTS);
			}
		}
		
		//Allocate arrays for activations:
		INDArray outputActivations = Nd4j.zeros(miniBatchSize,hiddenLayerSize,timeSeriesLength);
		INDArray rucZs = Nd4j.zeros(miniBatchSize,3*hiddenLayerSize,timeSeriesLength);	//zs for reset gate, update gate, candidate activation
		INDArray rucAs = Nd4j.zeros(miniBatchSize,3*hiddenLayerSize,timeSeriesLength);	//activations for above
		
		for( int t=0; t<timeSeriesLength; t++ ){
			INDArray prevLayerInputSlice = (is2dInput ? input : input.slice(t, 2));	//[Expected shape: [m,nIn]. Also deals with edge case of T=1, with 'time series' data of shape [m,nIn], equiv. to [m,nIn,1]
			INDArray prevOutputActivations = (t==0 ? Nd4j.zeros(miniBatchSize,hiddenLayerSize) : outputActivations.slice(t-1,2));	//Shape: [m,nL]
			
			//Calculate reset gate, update gate and candidate zs
			INDArray zs = prevLayerInputSlice.mmul(inputWeights)
					.addi(prevOutputActivations.mmul(recurrentWeights))
					.addiRowVector(biases);	//Shape: [m,3n^L]
			
			INDArray as = zs.dup();		//Want to apply sigmoid to both reset and update gates; user-settable activation on candidate activation
			Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid",
					as.get(NDArrayIndex.all(),NDArrayIndex.interval(0, 2*hiddenLayerSize))));
			Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(),
					as.get(NDArrayIndex.all(),NDArrayIndex.interval(2*hiddenLayerSize,3*hiddenLayerSize))));
			
			//Finally, calculate output activation:
			INDArray candidateAs = as.get(NDArrayIndex.all(),NDArrayIndex.interval(2*hiddenLayerSize,3*hiddenLayerSize));
			INDArray updateAs = as.get(NDArrayIndex.all(),NDArrayIndex.interval(hiddenLayerSize, 2*hiddenLayerSize));	//from {a_r, a_u, a_c} with shape [m,3*n^L]
			INDArray oneMinUpdateAs = updateAs.rsub(1);
			INDArray outputASlice = updateAs.mul(prevOutputActivations).addi(oneMinUpdateAs.muli(candidateAs));
			
			
			rucZs.slice(t,2).assign(zs);
			rucAs.slice(t,2).assign(as);
			outputActivations.slice(t,2).assign(outputASlice);
		}
		
		return new INDArray[]{outputActivations,rucZs,rucAs};
	}
	
	@Override
	public INDArray activationMean(){
		return activate();
	}

	@Override
	public Type type(){
		return Type.RECURRENT;
	}

	@Override
	public Layer transpose(){
		throw new UnsupportedOperationException("Not yet implemented");
	}
	
	@Override
    public double calcL2() {
    	if(!conf.isUseRegularization() || conf.getL2() <= 0.0 ) return 0.0;
    	double l2 = Transforms.pow(getParam(GRUParamInitializer.RECURRENT_WEIGHTS), 2).sum(Integer.MAX_VALUE).getDouble(0)
    			+ Transforms.pow(getParam(GRUParamInitializer.INPUT_WEIGHTS), 2).sum(Integer.MAX_VALUE).getDouble(0);
    	return 0.5 * conf.getL2() * l2;
    }

    @Override
    public double calcL1() {
    	if(!conf.isUseRegularization() || conf.getL1() <= 0.0 ) return 0.0;
        double l1 = Transforms.abs(getParam(GRUParamInitializer.RECURRENT_WEIGHTS)).sum(Integer.MAX_VALUE).getDouble(0)
        		+ Transforms.abs(getParam(GRUParamInitializer.INPUT_WEIGHTS)).sum(Integer.MAX_VALUE).getDouble(0);
        return conf.getL1() * l1;
    }
	
}
