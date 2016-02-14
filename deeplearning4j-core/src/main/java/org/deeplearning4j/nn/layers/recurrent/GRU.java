/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.GRUParamInitializer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/** Gated Recurrent Unit RNN Layer.<br>
 * The GRU was recently proposed by Cho et al. 2014 - http://arxiv.org/abs/1406.1078<br>
 * It is similar to the LSTM architecture in that both use a gating structure within each unit
 * to attempt to capture long-term dependencies and deal with the vanishing gradient problem.
 * A GRU layer contains fewer parameters than an equivalent size LSTM layer, and some research
 * (such as http://arxiv.org/abs/1412.3555) suggests it may outperform LSTM layers (given an
 * equal number of parameters) in some cases.
 * @author Alex Black
 */
public class GRU extends BaseRecurrentLayer<org.deeplearning4j.nn.conf.layers.GRU> {
	public static final String STATE_KEY_PREV_ACTIVATION = "prevAct";

	public GRU(NeuralNetConfiguration conf) {
		super(conf);
		throw new UnsupportedOperationException("GRU layer disabled: Backprop implementation is incorrect in this version. Consider using GravesLSTM instead");
	}
	
	public GRU(NeuralNetConfiguration conf, INDArray input) {
		super(conf, input);
		throw new UnsupportedOperationException("GRU layer disabled: Backprop implementation is incorrect in this version. Consider using GravesLSTM instead");
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
		//First: Do forward pass to get gate activations etc.
		INDArray[] activations = activateHelper(true,null);	//Order: {outputActivations,rucZs,rucAs}
		INDArray outputActivations = activations[0];
		INDArray rucZs = activations[1];
		INDArray rucAs = activations[2];
		
		INDArray inputWeights = getParam(GRUParamInitializer.INPUT_WEIGHT_KEY); //Shape: [n^(L-1),3*n^L], order: [wr,wu,wc]
		INDArray recurrentWeights = getParam(GRUParamInitializer.RECURRENT_WEIGHT_KEY);	//Shape: [n^L,3*n^L]; order: [wR,wU,wC]
		
		int layerSize = recurrentWeights.size(0);	//i.e., n^L
		int prevLayerSize = inputWeights.size(0);	//n^(L-1)
		int miniBatchSize = epsilon.size(0);
		boolean is2dInput = epsilon.rank() < 3; //Edge case: T=1 may have shape [miniBatchSize,n^(L+1)], equiv. to [miniBatchSize,n^(L+1),1]
		int timeSeriesLength = (is2dInput? 1: epsilon.size(2));
		
		INDArray wr = inputWeights.get(NDArrayIndex.all(),interval(0,layerSize));
		INDArray wu = inputWeights.get(NDArrayIndex.all(),interval(layerSize,2*layerSize));
		INDArray wc = inputWeights.get(NDArrayIndex.all(),interval(2*layerSize,3*layerSize));
		INDArray wR = recurrentWeights.get(NDArrayIndex.all(),interval(0,layerSize));
		INDArray wU = recurrentWeights.get(NDArrayIndex.all(),interval(layerSize,2*layerSize));
		INDArray wC = recurrentWeights.get(NDArrayIndex.all(),interval(2*layerSize,3*layerSize));
		INDArray wRdiag = Nd4j.diag(wR).transpose();
//		INDArray wUdiag = Nd4j.diag(wU).transpose();
		INDArray wCdiag = Nd4j.diag(wC).transpose();
		
		//Parameter gradients: Stores sum over each time step here
		INDArray biasGradients = Nd4j.zeros(new int[]{1,3*layerSize});
		INDArray inputWeightGradients = Nd4j.zeros(new int[]{prevLayerSize,3*layerSize});
		INDArray recurrentWeightGradients = Nd4j.zeros(new int[]{layerSize,3*layerSize});
		
		INDArray epsilonNext = Nd4j.zeros(miniBatchSize,prevLayerSize,timeSeriesLength);	//i.e., what would be W^L*(delta^L)^T. Shape: [m,n^(L-1),T]
		
		INDArray deltaOutNext = Nd4j.zeros(miniBatchSize,layerSize);
		for( int t=timeSeriesLength-1; t>=0; t-- ){
			INDArray prevOut = (t==0 ? Nd4j.zeros(miniBatchSize,layerSize) : outputActivations.tensorAlongDimension(t-1,1,0));	//Shape: [m,n^L]
			
			INDArray aSlice = (is2dInput ? rucAs : rucAs.tensorAlongDimension(t,1,0));
			INDArray zSlice = (is2dInput ? rucZs : rucZs.tensorAlongDimension(t,1,0));
			INDArray aSliceNext;
			INDArray zSliceNext;
			if(t == timeSeriesLength-1){
				aSliceNext = Nd4j.zeros(miniBatchSize,3*layerSize);
				zSliceNext = Nd4j.zeros(miniBatchSize,3*layerSize);
			} else {
				aSliceNext = rucAs.tensorAlongDimension(t+1,1,0);
				zSliceNext = rucZs.tensorAlongDimension(t+1,1,0);
			}

			INDArray zr = zSlice.get(NDArrayIndex.all(),interval(0,layerSize));
			INDArray sigmaPrimeZr = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", zr.dup()).derivative());
			
			INDArray epsilonSlice = (is2dInput ? epsilon : epsilon.tensorAlongDimension(t,1,0));		//(w^{L+1}*(delta^{(L+1)t})^T)^T or equiv.
			INDArray deltaOut = epsilonSlice.dup();
			if( t < timeSeriesLength-1 ){
				INDArray aOut = (is2dInput ? outputActivations : outputActivations.tensorAlongDimension(t,1,0));
				INDArray arNext = aSliceNext.get(NDArrayIndex.all(),interval(0,layerSize));
				INDArray auNext = aSliceNext.get(NDArrayIndex.all(),interval(layerSize,2*layerSize));
				INDArray acNext = aSliceNext.get(NDArrayIndex.all(),interval(2*layerSize,3*layerSize));
				INDArray zrNext = zSliceNext.get(NDArrayIndex.all(),interval(0,layerSize));
				INDArray zuNext = zSliceNext.get(NDArrayIndex.all(),interval(layerSize,2*layerSize));
				INDArray zcNext = zSliceNext.get(NDArrayIndex.all(),interval(2*layerSize,3*layerSize));
				
				INDArray sigmaPrimeZrNext = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", zrNext.dup()).derivative());
				INDArray sigmaPrimeZuNext = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", zuNext.dup()).derivative());
				INDArray sigmaPrimeZcNext = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), zcNext.dup()).derivative());
				
				deltaOut.addi(auNext.mul(deltaOutNext));
				deltaOut.addi(aOut.sub(acNext).muli(sigmaPrimeZuNext).muli( wU.mmul(deltaOutNext.transpose()).transpose() ) );
				deltaOut.addi(auNext.rsub(1.0)
						.muli( sigmaPrimeZcNext )
						.muli( arNext.add(aOut.mul(sigmaPrimeZrNext).muliRowVector(wRdiag)) )
						.muli( wC.mmul(deltaOutNext.transpose()).transpose() )
						);
			}
			
			//Delta at update gate
			INDArray zu = zSlice.get(NDArrayIndex.all(),interval(layerSize,2*layerSize));
			INDArray sigmaPrimeZu = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", zu.dup()).derivative());
			INDArray ac = aSlice.get(NDArrayIndex.all(),interval(2*layerSize,3*layerSize));
			INDArray deltaU = deltaOut.mul(sigmaPrimeZu).muli(prevOut.sub(ac));
			
			//Delta for candidate activation
			INDArray zc = zSlice.get(NDArrayIndex.all(),interval(2*layerSize,3*layerSize));
			INDArray sigmaPrimeZc = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), zc.dup()).derivative());
			INDArray au = aSlice.get(NDArrayIndex.all(),interval(layerSize,2*layerSize));
			INDArray deltaC = deltaOut.mul(sigmaPrimeZc).muli(au.rsub(1.0));
			
			//Delta at reset gate
			INDArray deltaR = deltaC.mulRowVector(wCdiag).muli(prevOut).muli(sigmaPrimeZr);
			
			//Add input gradients for this time step:
			INDArray prevLayerActivationSlice = (is2dInput ? input : input.tensorAlongDimension(t,1,0));
			inputWeightGradients.get(NDArrayIndex.all(),interval(0,layerSize))
				.addi(deltaR.transpose().mmul(prevLayerActivationSlice).transpose());
			inputWeightGradients.get(NDArrayIndex.all(),interval(layerSize,2*layerSize))
				.addi(deltaU.transpose().mmul(prevLayerActivationSlice).transpose());
			inputWeightGradients.get(NDArrayIndex.all(),interval(2*layerSize,3*layerSize))
				.addi(deltaC.transpose().mmul(prevLayerActivationSlice).transpose());
			
			//Add recurrent weight gradients for this time step:
			if(t>0){	//t=0: no previous output
				recurrentWeightGradients.get(NDArrayIndex.all(),interval(0,layerSize))
					.addi(deltaR.transpose().mmul(prevOut).transpose());
				recurrentWeightGradients.get(NDArrayIndex.all(),interval(layerSize,2*layerSize))
					.addi(deltaU.transpose().mmul(prevOut).transpose());
				INDArray ar = aSlice.get(NDArrayIndex.all(),interval(0,layerSize));
				recurrentWeightGradients.get(NDArrayIndex.all(),interval(2*layerSize,3*layerSize))
					.addi(deltaC.transpose().mmul(prevOut.mul(ar)).transpose());
			}
			
			//Add bias gradients for this time step:
			biasGradients.get(NDArrayIndex.point(0),interval(0,layerSize)).addi(deltaR.sum(0));
			biasGradients.get(NDArrayIndex.point(0),interval(layerSize,2*layerSize)).addi(deltaU.sum(0));
			biasGradients.get(NDArrayIndex.point(0),interval(2*layerSize,3*layerSize)).addi(deltaC.sum(0));

			INDArray epsilonNextSlice = wr.mmul(deltaR.transpose()).transpose()
					.addi(wu.mmul(deltaU.transpose()).transpose())
					.addi(wc.mmul(deltaC.transpose()).transpose());
			epsilonNext.tensorAlongDimension(t,1,0).assign(epsilonNextSlice);
			
			deltaOutNext = deltaOut;
		}
		
		Gradient g = new DefaultGradient();
		g.setGradientFor(GRUParamInitializer.INPUT_WEIGHT_KEY, inputWeightGradients);
		g.setGradientFor(GRUParamInitializer.RECURRENT_WEIGHT_KEY,recurrentWeightGradients);
		g.setGradientFor(GRUParamInitializer.BIAS_KEY, biasGradients);
		
		return new Pair<>(g,epsilonNext);
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
		setInput(input);
		return activateHelper(training,null)[0];
	}

	@Override
	public INDArray activate(INDArray input){
		setInput(input);
		return activateHelper(true,null)[0];
	}

	@Override
	public INDArray activate(boolean training){
		return activateHelper(training,null)[0];
	}

	@Override
	public INDArray activate(){
		return activateHelper(false,null)[0];
	}

	/** Returns activations array: {output,rucZs,rucAs} in that order. */
	private INDArray[] activateHelper(boolean training, INDArray prevOutputActivations){
		
		INDArray inputWeights = getParam(GRUParamInitializer.INPUT_WEIGHT_KEY); //Shape: [n^(L-1),3*n^L], order: [wr,wu,wc]
		INDArray recurrentWeights = getParam(GRUParamInitializer.RECURRENT_WEIGHT_KEY);	//Shape: [n^L,3*n^L]; order: [wR,wU,wC]
		INDArray biases = getParam(GRUParamInitializer.BIAS_KEY); //Shape: [1,3*n^L]; order: [br,bu,bc]
		
		
		
		boolean is2dInput = input.rank() < 3;		//Edge case of T=1, may have shape [m,nIn], equiv. to [m,nIn,1]
		int timeSeriesLength = (is2dInput ? 1 : input.size(2));
		int hiddenLayerSize = recurrentWeights.size(0);
		int miniBatchSize = input.size(0);
		
		int layerSize = hiddenLayerSize;
		INDArray wr = inputWeights.get(NDArrayIndex.all(),interval(0,layerSize));
		INDArray wu = inputWeights.get(NDArrayIndex.all(),interval(layerSize,2*layerSize));
		INDArray wc = inputWeights.get(NDArrayIndex.all(),interval(2*layerSize,3*layerSize));
		INDArray wR = recurrentWeights.get(NDArrayIndex.all(),interval(0,layerSize));
		INDArray wU = recurrentWeights.get(NDArrayIndex.all(),interval(layerSize,2*layerSize));
		INDArray wC = recurrentWeights.get(NDArrayIndex.all(),interval(2*layerSize,3*layerSize));
		INDArray br = biases.get(NDArrayIndex.point(0),interval(0,layerSize));
		INDArray bu = biases.get(NDArrayIndex.point(0),interval(layerSize,2*layerSize));
		INDArray bc = biases.get(NDArrayIndex.point(0),interval(2*layerSize,3*layerSize));
//		INDArray wRAndU = recurrentWeights.get(NDArrayIndex.all(),NDArrayIndex.interval(0, 2*hiddenLayerSize));
//		INDArray wC = recurrentWeights.get(NDArrayIndex.all(),NDArrayIndex.interval(2*hiddenLayerSize,3*hiddenLayerSize));
		
		//Apply dropconnect to input (not recurrent) weights only:
		if(conf.isUseDropConnect() && training) {
			if (conf.getLayer().getDropOut() > 0) {
				inputWeights = Dropout.applyDropConnect(this,GRUParamInitializer.INPUT_WEIGHT_KEY);
			}
		}
		
		//Allocate arrays for activations:
		INDArray outputActivations = Nd4j.zeros(miniBatchSize,hiddenLayerSize,timeSeriesLength);
		INDArray rucZs = Nd4j.zeros(miniBatchSize,3*hiddenLayerSize,timeSeriesLength);	//zs for reset gate, update gate, candidate activation
		INDArray rucAs = Nd4j.zeros(miniBatchSize,3*hiddenLayerSize,timeSeriesLength);	//activations for above
		
		if(prevOutputActivations==null) prevOutputActivations = Nd4j.zeros(miniBatchSize,hiddenLayerSize);
		for( int t=0; t<timeSeriesLength; t++ ){
			INDArray prevLayerInputSlice = (is2dInput ? input : input.tensorAlongDimension(t,1,0));	//[Expected shape: [m,nIn]. Also deals with edge case of T=1, with 'time series' data of shape [m,nIn], equiv. to [m,nIn,1]
			if(t>0) prevOutputActivations = outputActivations.tensorAlongDimension(t-1,1,0); //Shape: [m,nL]
			
			/* This commented out implementation: should be same as 'naive' implementation that follows.
			 * Using naive approach at present for debugging purposes
			 * 
			//Calculate reset gate, update gate and candidate zs
				//First: inputs + biases for all (reset gate, update gate, candidate activation)
			INDArray zs = prevLayerInputSlice.mmul(inputWeights).addiRowVector(biases);	//Shape: [m,3n^L]
			
			//Recurrent weights * prevInput for reset and update gates:
			INDArray zrAndu = zs.get(NDArrayIndex.all(),NDArrayIndex.interval(0, 2*hiddenLayerSize));
			zrAndu.addi(prevOutputActivations.mmul(wRAndU));	//zr and zu now have all components
			
			INDArray as = zs.dup();
			INDArray arAndu = as.get(NDArrayIndex.all(),NDArrayIndex.interval(0, 2*hiddenLayerSize));
			Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", arAndu));	//Sigmoid for both reset and update gates
			
			//Recurrent component of candidate z: (previously: zc has only input and bias components)
			INDArray ar = as.get(NDArrayIndex.all(),NDArrayIndex.interval(0, hiddenLayerSize));
			INDArray zc = zs.get(NDArrayIndex.all(),NDArrayIndex.interval(2*hiddenLayerSize, 3*hiddenLayerSize));
			zc.addi(ar.mul(prevOutputActivations).mmul(wC));
			
			INDArray ac = as.get(NDArrayIndex.all(),NDArrayIndex.interval(2*hiddenLayerSize, 3*hiddenLayerSize));
			ac.assign(zc);
			Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(),ac));
			
			//Finally, calculate output activation:
			INDArray au = as.get(NDArrayIndex.all(),NDArrayIndex.interval(hiddenLayerSize, 2*hiddenLayerSize));
			INDArray outputASlice = au.mul(prevOutputActivations).addi(au.rsub(1).muli(ac));
			*/
			
			INDArray zs = Nd4j.zeros(miniBatchSize,3*hiddenLayerSize);
			INDArray as = Nd4j.zeros(miniBatchSize,3*hiddenLayerSize);
			
			INDArray zr = prevLayerInputSlice.mmul(wr).addi(prevOutputActivations.mmul(wR)).addiRowVector(br);
			INDArray ar = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid",zr.dup()));
			zs.get(NDArrayIndex.all(),NDArrayIndex.interval(0, hiddenLayerSize)).assign(zr);
			as.get(NDArrayIndex.all(),NDArrayIndex.interval(0, hiddenLayerSize)).assign(ar);
			
			INDArray zu = prevLayerInputSlice.mmul(wu).addi(prevOutputActivations.mmul(wU)).addiRowVector(bu);
			INDArray au = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid",zu.dup()));
			zs.get(NDArrayIndex.all(),NDArrayIndex.interval(hiddenLayerSize, 2*hiddenLayerSize)).assign(zu);
			as.get(NDArrayIndex.all(),NDArrayIndex.interval(hiddenLayerSize, 2*hiddenLayerSize)).assign(au);
			
			INDArray zc = prevLayerInputSlice.mmul(wc).addi(prevOutputActivations.mul(ar).mmul(wC)).addiRowVector(bc);
			INDArray ac = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(),zc.dup()));
			zs.get(NDArrayIndex.all(),NDArrayIndex.interval(2*hiddenLayerSize, 3*hiddenLayerSize)).assign(zc);
			as.get(NDArrayIndex.all(),NDArrayIndex.interval(2*hiddenLayerSize, 3*hiddenLayerSize)).assign(ac);
			
			INDArray aOut = au.mul(prevOutputActivations).addi(au.rsub(1).mul(ac));
			
			
			rucZs.tensorAlongDimension(t,1,0).assign(zs);
			rucAs.tensorAlongDimension(t,1,0).assign(as);
			outputActivations.tensorAlongDimension(t,1,0).assign(aOut);
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
    	if(!conf.isUseRegularization() || conf.getLayer().getL2() <= 0.0 ) return 0.0;

		double l2Norm = getParam(GRUParamInitializer.RECURRENT_WEIGHT_KEY).norm2Number().doubleValue();
		double sumSquaredWeights = l2Norm*l2Norm;

		l2Norm = getParam(GRUParamInitializer.INPUT_WEIGHT_KEY).norm2Number().doubleValue();
		sumSquaredWeights += l2Norm*l2Norm;

		return 0.5 * conf.getLayer().getL2() * sumSquaredWeights;
    }

    @Override
    public double calcL1() {
    	if(!conf.isUseRegularization() || conf.getLayer().getL1() <= 0.0 ) return 0.0;
        double l1 = Transforms.abs(getParam(GRUParamInitializer.RECURRENT_WEIGHT_KEY)).sum(Integer.MAX_VALUE).getDouble(0)
        		+ Transforms.abs(getParam(GRUParamInitializer.INPUT_WEIGHT_KEY)).sum(Integer.MAX_VALUE).getDouble(0);
        return conf.getLayer().getL1() * l1;
    }

	@Override
	public INDArray rnnTimeStep(INDArray input) {
		setInput(input);
		INDArray[] activations = activateHelper(false,stateMap.get(STATE_KEY_PREV_ACTIVATION));
		INDArray outAct = activations[0];
		//Store last time step of output activations for later use:
		int tLength = outAct.size(2);
		INDArray lastActSlice = outAct.tensorAlongDimension(tLength-1,1,0);
		stateMap.put(STATE_KEY_PREV_ACTIVATION, lastActSlice.dup());

		return outAct;
	}

	@Override
	public INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT) {
		setInput(input);
		INDArray[] activations = activateHelper(false,stateMap.get(STATE_KEY_PREV_ACTIVATION));
		INDArray outAct = activations[0];
		if(storeLastForTBPTT){
			//Store last time step of output activations for later use:
			int tLength = outAct.size(2);
			INDArray lastActSlice = outAct.tensorAlongDimension(tLength-1,1,0);
			tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, lastActSlice.dup());
		}

		return outAct;
	}

	@Override
	public Pair<Gradient, INDArray> tbpttBackpropGradient(INDArray epsilon, int tbpttBackwardLength){
		throw new UnsupportedOperationException("Not yet implemented");
	}
}
