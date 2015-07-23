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
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 *
 * LSTM layer implementation.
 * Based on Graves: Supervised Sequence Labelling with Recurrent Neural Networks
 * http://www.cs.toronto.edu/~graves/phd.pdf
 * See also for full/vectorized equations (and a comparison to other LSTM variants):
 * Greff et al. 2015, "LSTM: A Search Space Odyssey", pg11. This is the "vanilla" variant in said paper
 * http://arxiv.org/pdf/1503.04069.pdf
 *
 *
 * @author Alex Black
 */
public class GravesLSTM extends BaseLayer {
	private static final long serialVersionUID = 4115420413387754109L;

	public GravesLSTM(NeuralNetConfiguration conf) {
		super(conf);
	}

	@Override
	public INDArray transform(INDArray data) {
		return activate(data);
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
	public Gradient backpropGradient(Gradient gradient, Layer layers) {
		//First: Do forward pass to get gate activations etc.
		INDArray[] activations = activateHelper(input(), true);	//Order: {outputActivations,memCellActivations,ifogZs,ifogAs}
		INDArray outputActivations = activations[0];
		INDArray memCellActivations = activations[1];
		INDArray ifogZs = activations[2];
		INDArray ifogAs = activations[3];
		
		INDArray inputWeights = getParam(GravesLSTMParamInitializer.INPUT_WEIGHTS);
		INDArray recurrentWeights = getParam(GravesLSTMParamInitializer.RECURRENT_WEIGHTS);	//Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
		
		//Expect errors to have shape: [miniBatchSize,n^(L+1),timeSeriesLength]
		int hiddenLayerSize = recurrentWeights.rows();	//i.e., n^L
		int prevLayerSize = getParam(GravesLSTMParamInitializer.INPUT_WEIGHTS).shape()[0];
		int miniBatchSize = nextDelta.size(0);
		boolean is2dInput = nextDelta.rank() < 3; //Edge case: T=1 may have shape [miniBatchSize,n^(L+1)], equiv. to [miniBatchSize,n^(L+1),1]
		int timeSeriesLength = (is2dInput? 1: nextDelta.size(2));
		
		INDArray wi = inputWeights.get(interval(0,prevLayerSize),interval(0,hiddenLayerSize));
		INDArray wI = recurrentWeights.get(interval(0,hiddenLayerSize),interval(0,hiddenLayerSize));
		INDArray wf = inputWeights.get(interval(0,prevLayerSize),interval(hiddenLayerSize,2*hiddenLayerSize));
		INDArray wF = recurrentWeights.get(interval(0,hiddenLayerSize),interval(hiddenLayerSize,2*hiddenLayerSize));
		INDArray wFF = recurrentWeights.get(interval(0,hiddenLayerSize),interval(4*hiddenLayerSize,4*hiddenLayerSize+1));
		INDArray wo = inputWeights.get(interval(0,prevLayerSize),interval(2*hiddenLayerSize,3*hiddenLayerSize));
		INDArray wO = recurrentWeights.get(interval(0,hiddenLayerSize),interval(2*hiddenLayerSize,3*hiddenLayerSize));
		INDArray wOO = recurrentWeights.get(interval(0,hiddenLayerSize),interval(4*hiddenLayerSize+1,4*hiddenLayerSize+2));
		INDArray wg = inputWeights.get(interval(0,prevLayerSize),interval(3*hiddenLayerSize,4*hiddenLayerSize));
		INDArray wG = recurrentWeights.get(interval(0,hiddenLayerSize),interval(3*hiddenLayerSize,4*hiddenLayerSize));
		INDArray wGG = recurrentWeights.get(interval(0,hiddenLayerSize),interval(4*hiddenLayerSize+2,4*hiddenLayerSize+3));

		//Gradient arrays. Note these are pre-sum; so need to sum along time (and along mini-batch for biases)
		INDArray biasGradients = Nd4j.zeros(new int[]{miniBatchSize,4*hiddenLayerSize,timeSeriesLength});
		INDArray inputWeightGradients = Nd4j.zeros(new int[]{prevLayerSize,4*hiddenLayerSize,timeSeriesLength});
		INDArray recurrentWeightGradients = Nd4j.zeros(new int[]{hiddenLayerSize,4*hiddenLayerSize+3,timeSeriesLength});	//Order: {I,F,O,G,FF,OO,GG}
		
		INDArray epsilonNext = Nd4j.zeros(miniBatchSize,prevLayerSize,timeSeriesLength);	//i.e., what would be W^L*(delta^L)^T. Shape: [m,n^(L-1),T]
		
		/*Placeholder. To be replaced by masking array for used for variable length time series
		 *Idea: M[i,j] = 1 if data is present for time j in example i in mini-batch.
		 *M[i,j] = 0 otherwise
		 *Then do a column multiply to set appropriate deltas to 0 if data is beyond end of time series
		 *for the corresponding example
		 */
//		INDArray timeSeriesMaskArray = Nd4j.ones(miniBatchSize,timeSeriesLength);	//For now: assume that all data in mini-batch is of length 'timeSeriesLength'

		for( int t=timeSeriesLength-1; t>=0; t-- ){
			INDArray prevMemCellActivations = (t==0 ? Nd4j.zeros(hiddenLayerSize, hiddenLayerSize) : memCellActivations.slice(t-1, 2) );	//Shape: [n^L, n^L]
			INDArray prevHiddenUnitActivation = (t==0 ? Nd4j.zeros(hiddenLayerSize, hiddenLayerSize) : outputActivations.slice(t-1,2) );	//Shape: [n^L, n^L]; i.e., layer output at prev. time step.

			INDArray nextLayerDeltaSlice = nextDelta;	//delta^{(L+1)t}
			if (!is2dInput) {
				nextLayerDeltaSlice = nextDelta.slice(t, 2);
			}
					
			//delta_i^{L(t+1)}
			INDArray deltaiNext = (t==timeSeriesLength-1 ?
					Nd4j.zeros(miniBatchSize,hiddenLayerSize) :
					biasGradients.slice(t+1,2).get(new NDArrayIndex[]{interval(0,miniBatchSize),interval(0,hiddenLayerSize)}));
			//delta_f^{L(t+1)}
			INDArray deltafNext = (t==timeSeriesLength-1 ?
					Nd4j.zeros(miniBatchSize,hiddenLayerSize) :
					biasGradients.slice(t+1,2).get(new NDArrayIndex[]{interval(0,miniBatchSize),interval(hiddenLayerSize,2*hiddenLayerSize)}));
			//delta_o^{L(t+1)}
			INDArray deltaoNext = (t==timeSeriesLength-1 ?
					Nd4j.zeros(miniBatchSize,hiddenLayerSize) :
					biasGradients.slice(t+1,2).get(new NDArrayIndex[]{interval(0,miniBatchSize),interval(2*hiddenLayerSize,3*hiddenLayerSize)}));
			//delta_g^{L(t+1)}
			INDArray deltagNext = (t==timeSeriesLength-1 ?
					Nd4j.zeros(miniBatchSize,hiddenLayerSize) :
					biasGradients.slice(t+1,2).get(new NDArrayIndex[]{interval(0,miniBatchSize),interval(3*hiddenLayerSize,4*hiddenLayerSize)}));

			//For variable length mini-batch data: Zero out deltas as necessary, so deltas beyond end of each time series are always 0
			//Not implemented yet, but left here for when this is implemented
			/*
			if( t < timeSeriesLength-1 ){
				INDArray maskColumn = timeSeriesMaskArray.getColumn(t);
				deltaiNext.muliColumnVector(maskColumn);
				deltafNext.muliColumnVector(maskColumn);
				deltaoNext.muliColumnVector(maskColumn);
				deltagNext.muliColumnVector(maskColumn);
			}*/

			//LSTM unit output errors (dL/d(a_out)); not to be confused with \delta=dL/d(z_out)
			//INDArray nablaOut = nextLayerDeltaSlice.mmul(nextWeights.transpose())
			INDArray epsilonSlice = epsilon.slice(t, 2);		//(w^{L+1}*(delta^{(L+1)t})^T)^T
			INDArray nablaOut = epsilonSlice
					.addi(deltaiNext.mmul(wI.transpose()))
					.addi(deltafNext.mmul(wF.transpose()))
					.addi(deltaoNext.mmul(wO.transpose()))
					.addi(deltagNext.mmul(wG.transpose()));
			//Shape: [m,n^L]

			//Output gate deltas:
			INDArray sigmahOfS = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(), prevMemCellActivations.dup()));//	shape: [m,n^L]
			INDArray zo = ifogZs.slice(t, 2).get(NDArrayIndex.all(),interval(2*hiddenLayerSize,3*hiddenLayerSize));
			INDArray sigmaoPrimeOfZo = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", zo).derivative());//			shape: [m,n^L]
			INDArray deltao = nablaOut.mul(sigmahOfS).muli(sigmaoPrimeOfZo);
			//Shape: [m,n^L]

			//Memory cell error:
			INDArray sigmahPrimeOfS = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(), prevMemCellActivations.dup()));//	shape: [m,n^L]
			INDArray nextForgetGateAs = (t==timeSeriesLength-1 ? Nd4j.zeros(miniBatchSize,hiddenLayerSize) :
					ifogAs.slice(t,2).get(NDArrayIndex.all(),interval(2 * hiddenLayerSize,3*hiddenLayerSize)) );
			INDArray nablaCellState = nablaOut.mul(prevHiddenUnitActivation).muli(sigmahPrimeOfS)
					.addi(nextForgetGateAs.mul(prevMemCellActivations))
					.addi(deltafNext.mmul(Nd4j.diag(wFF)))
					.addi(deltaoNext.mmul(Nd4j.diag(wOO)))
					.addi(deltagNext.mmul(Nd4j.diag(wGG)));

			//Forget gate delta:
			INDArray zf = ifogZs.slice(t, 0).get(NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize));	//z_f^{Lt}	shape: [m,n^L]
			INDArray deltaf = nablaCellState.mul(prevMemCellActivations)
					.muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", zf).derivative()));
			//Shape: [m,n^L]

			//Input modulation gate delta:
			INDArray zg = ifogZs.slice(t, 0).get(NDArrayIndex.all(),interval(3*hiddenLayerSize,4 * hiddenLayerSize));	//z_g^{Lt}	shape: [m,n^L]
			INDArray ai = ifogAs.slice(t, 0).get(NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize));	//a_i^{Lt}	shape: [m,n^L]
			INDArray deltag = nablaCellState.mul(ai)
					.muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", zg).derivative()));
			//Shape: [m,n^L]

			//Network input delta:
			INDArray zi = ifogZs.slice(t, 0).get(NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize));	//z_i^{Lt}	shape: [m,n^L]
			INDArray ag = ifogAs.slice(t, 0).get(NDArrayIndex.all(),interval(3*hiddenLayerSize,4 * hiddenLayerSize));	//a_g^{Lt}	shape: [m,n^L]
			INDArray deltai = nablaCellState.mul(ag)
					.muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("tanh", zi).derivative()));
				//Shape: [m,n^L]
			
			INDArray prevLayerActivationSlice = input.slice(t, 2);
			//Indexing here: all columns (==interval(0,n^(L-1)), 3rd dimension based on IFOG order. Sum over mini-batches occurs in delta*prevLayerActivations
			inputWeightGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),interval(0,hiddenLayerSize)}, deltai.transpose().mmul(prevLayerActivationSlice).transpose());
			inputWeightGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize)}, deltaf.transpose().mmul(prevLayerActivationSlice).transpose());
			inputWeightGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),interval(2 * hiddenLayerSize,3 * hiddenLayerSize)}, deltao.transpose().mmul(prevLayerActivationSlice).transpose());
			inputWeightGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),interval(3 * hiddenLayerSize,4 * hiddenLayerSize)}, deltag.transpose().mmul(prevLayerActivationSlice).transpose());

			if( t > 0 ){
				//Minor optimization. If t==0, then prevHiddenUnitActivation==zeros(n^L,n^L), so dL/dW for recurrent weights will end up as 0 anyway. (They are initialized as 0)
				recurrentWeightGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),interval(0,hiddenLayerSize)}, deltai.transpose().mmul(prevHiddenUnitActivation).transpose());	//dL/dw_{Ixy} = delta_{ix} * a_{iy}^{L(t-1)}
				recurrentWeightGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize)}, deltaf.transpose().mmul(prevHiddenUnitActivation).transpose());	//dL/dw_{Fxy} = delta_{fx} * a_{iy}^{L(t-1)}
				recurrentWeightGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),interval(2 * hiddenLayerSize,3 * hiddenLayerSize)}, deltao.transpose().mmul(prevHiddenUnitActivation).transpose());	//dL/dw_{O}
				recurrentWeightGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),interval(3 * hiddenLayerSize,4 * hiddenLayerSize)}, deltag.transpose().mmul(prevHiddenUnitActivation).transpose());	//dL/dw_{O}

				INDArray dLdwFF = deltaf.mul(prevMemCellActivations);	//mul not mmul because these weights are from unit j->j only (whereas other recurrent weights are i->j for all i,j)
				recurrentWeightGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),new NDArrayIndex(4*hiddenLayerSize)}, dLdwFF);	//dL/dw_{FF}
				INDArray dLdwGG = deltag.mul(prevMemCellActivations);
				recurrentWeightGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),new NDArrayIndex(4*hiddenLayerSize + 2)}, dLdwGG);	//dL/dw_{GG}
			}
			INDArray dLdwOO = deltao.transpose().mul(memCellActivations.slice(t,2)).transpose();
			recurrentWeightGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),new NDArrayIndex(4*hiddenLayerSize + 1)}, dLdwOO);	//dL/dw_{OOxy}

			if( miniBatchSize == 1 ){
				//Mini-batch size = 1 -> nRows = 1 -> special case for indexing...
				biasGradients.slice(t,2).put(new NDArrayIndex[]{interval(0,hiddenLayerSize)}, deltai);
				biasGradients.slice(t,2).put(new NDArrayIndex[]{interval(hiddenLayerSize,2*hiddenLayerSize)}, deltaf);
				biasGradients.slice(t,2).put(new NDArrayIndex[]{interval(2*hiddenLayerSize,3*hiddenLayerSize)}, deltao);
				biasGradients.slice(t,2).put(new NDArrayIndex[]{interval(3*hiddenLayerSize,4*hiddenLayerSize)}, deltag);
			} else {
				biasGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),interval(0,hiddenLayerSize)}, deltai);
				biasGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize)}, deltaf);
				biasGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),interval(2*hiddenLayerSize,3 * hiddenLayerSize)}, deltao);
				biasGradients.slice(t,2).put(new NDArrayIndex[]{NDArrayIndex.all(),interval(3*hiddenLayerSize,4 * hiddenLayerSize)}, deltag);
			}
			// TODO potential issue...
			//Calculate epsilonNext - i.e., equiv. to what would be (w^L*(d^(Lt))^T)^T in a normal network
			//But here, need to add 4 weights * deltas for the IFOG gates
			INDArray epsilonNextSlice = wi.mmul(deltai.transpose()).transpose()
					.addi(wf.mmul(deltaf.transpose()).transpose())
					.addi(wo.mmul(deltao.transpose()).transpose())
					.addi(wg.mmul(deltag.transpose()).transpose());
			epsilonNext.slice(t,2).assign(epsilonNextSlice);
		}

		//Weight/bias gradients: sum across time dimension. For bias gradients, sum across mini-batch also.
		Gradient ret = new DefaultGradient();
		ret.gradientForVariable().put(GravesLSTMParamInitializer.INPUT_WEIGHTS,inputWeightGradients.sum(2));
		ret.gradientForVariable().put(GravesLSTMParamInitializer.RECURRENT_WEIGHTS,recurrentWeightGradients.sum(2));
		ret.gradientForVariable().put(GravesLSTMParamInitializer.BIAS, biasGradients.sum(2).sum(0));
		
		return ret;
	}

	@Override
	public INDArray preOutput(INDArray x) {
		return activate(x,true);
	}

	@Override
	public INDArray activate(INDArray input, boolean training){
		return activateHelper(input, training)[0];
	}

	/**Returns 4 INDArrays: [outputActivations, memCellActivations, ifogZs, ifogAs] in that order.
	 * Need all 4 to do backward pass, but only care about the first one for forward pass.
	 */
	private INDArray[] activateHelper(INDArray input, boolean training){
		//Mini-batch data format: for mini-batch size m, nIn inputs, and T time series length
		//Data has shape [m,nIn,T]. Layer activations/output has shape [m,nHiddenUnits,T]

		INDArray recurrentWeights = getParam(GravesLSTMParamInitializer.RECURRENT_WEIGHTS);	//Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
		INDArray inputWeights = getParam(GravesLSTMParamInitializer.INPUT_WEIGHTS);			//Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
		INDArray biases = getParam(GravesLSTMParamInitializer.BIAS); //by row: IFOG			//Shape: [4,hiddenLayerSize]; order: [bi,bf,bo,bg]^T

		int[] dataShape = input.shape();
		boolean is2dInput = dataShape.length < 3;		//Edge case of T=1, may have shape [m,nIn], equiv. to [m,nIn,1]
		int timeSeriesLength = (is2dInput ? 1 : dataShape[2]);
		int hiddenLayerSize = recurrentWeights.rows();	//.shape()[0];
		int miniBatchSize = dataShape[0];
		int nIn = inputWeights.shape()[0];		//Size of previous layer (or input)

		//Apply dropconnect to input (not recurrent) weights only:
		if(conf.isUseDropConnect() && training) {
			if (conf.getDropOut() > 0) {
				inputWeights = Dropout.applyDropConnect(this,GravesLSTMParamInitializer.RECURRENT_WEIGHTS);
			}
		}

		//Extract weights and biases:
		INDArray wi = inputWeights.get(interval(0,nIn),interval(0,hiddenLayerSize));	//i.e., want rows 0..nIn, columns 0..hiddenLayerSize
		INDArray wI = recurrentWeights.get(interval(0,hiddenLayerSize),interval(0,hiddenLayerSize));
		INDArray bi = biases.get(interval(0,hiddenLayerSize));

		INDArray wf = inputWeights.get(interval(0,nIn),interval(hiddenLayerSize,2 * hiddenLayerSize));
		INDArray wF = recurrentWeights.get(interval(0,hiddenLayerSize),interval(hiddenLayerSize,2 * hiddenLayerSize));
		INDArray wFF = recurrentWeights.get(interval(0,hiddenLayerSize),interval(4*hiddenLayerSize,4 * hiddenLayerSize + 1));
		INDArray bf = biases.get(interval(hiddenLayerSize,2*hiddenLayerSize));

		INDArray wo = inputWeights.get(interval(0,nIn),interval(2 * hiddenLayerSize,3 * hiddenLayerSize));
		INDArray wO = recurrentWeights.get(interval(0,hiddenLayerSize),interval(2 * hiddenLayerSize,3 * hiddenLayerSize));
		INDArray wOO = recurrentWeights.get(interval(0,hiddenLayerSize),interval(4 * hiddenLayerSize + 1,4 * hiddenLayerSize + 2));
		INDArray bo = biases.get(interval(2*hiddenLayerSize,3 * hiddenLayerSize));

		INDArray wg = inputWeights.get(interval(0,nIn),interval(3 * hiddenLayerSize,4*hiddenLayerSize));
		INDArray wG = recurrentWeights.get(interval(0,hiddenLayerSize),interval(3*hiddenLayerSize,4 * hiddenLayerSize));
		INDArray wGG = recurrentWeights.get(interval(0,hiddenLayerSize),interval(4*hiddenLayerSize + 2,4 * hiddenLayerSize + 3));
		INDArray bg = biases.get(interval(3*hiddenLayerSize,4 * hiddenLayerSize));

		//Allocate arrays for activations:
		INDArray outputActivations = Nd4j.zeros(new int[]{miniBatchSize,hiddenLayerSize,timeSeriesLength});
		INDArray ifogZ = Nd4j.zeros(new int[]{miniBatchSize,4 * hiddenLayerSize,timeSeriesLength});
		INDArray ifogA = Nd4j.zeros(new int[]{miniBatchSize,4 * hiddenLayerSize,timeSeriesLength});
		INDArray memCellActivations = Nd4j.zeros(new int[]{miniBatchSize,hiddenLayerSize,timeSeriesLength});


		for( int t = 0; t < timeSeriesLength; t++ ){
			INDArray miniBatchData = (is2dInput ? input : input.slice(t, 2));	//[Expected shape: [m,nIn]. Also deals with edge case of T=1, with 'time series' data of shape [m,nIn], equiv. to [m,nIn,1]
			INDArray prevOutputActivations = (t==0 ? Nd4j.zeros(new int[]{miniBatchSize,hiddenLayerSize}) : outputActivations.slice(t-1,2));	//Shape: [m,nL]
			INDArray prevMemCellActivations = (t==0 ? Nd4j.zeros(new int[]{miniBatchSize,hiddenLayerSize}) : memCellActivations.slice(t-1,2));	//Shape: [m,nL]

			//Calculate activations for: network input + forget, output, input modulation gates.
			INDArray inputActivations = miniBatchData.mmul(wi)
					.addi(prevOutputActivations.mmul(wI))
					.addiRowVector(bi);
			NDArrayIndex[] iIndexes = (miniBatchSize == 1 ?
					new NDArrayIndex[]{interval(0,hiddenLayerSize)} :		//Indexing: special case for miniBatchSize=nRows=1. can't use "NDArrayIndex.all(),interval(0,hiddenLayerSize)"
					new NDArrayIndex[]{NDArrayIndex.all(),interval(0,hiddenLayerSize)} );
			ifogZ.slice(t,2).put(iIndexes, inputActivations);
			ifogA.slice(t,2).put(iIndexes, Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(), inputActivations)));


			INDArray forgetGateActivations = miniBatchData.mmul(wf)
					.addi(prevOutputActivations.mmul(wF))
					.addi(prevMemCellActivations.mmul(Nd4j.diag(wFF)))
					.addiRowVector(bf);
			NDArrayIndex[] fIndexes = (miniBatchSize == 1 ? new NDArrayIndex[]{interval(hiddenLayerSize,2*hiddenLayerSize)} :
					new NDArrayIndex[]{NDArrayIndex.all(),interval(hiddenLayerSize,2*hiddenLayerSize)});
			ifogZ.slice(t,2).put(fIndexes, forgetGateActivations);
			ifogA.slice(t,2).put(fIndexes, Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", forgetGateActivations)));
			//Reason for diag above: convert column vector -> diagonal matrix. Cell activations are only connected to the FOG gates in the same unit.
			//They are not connected to any other unit -> wFF_ij = 0 for i \neq j

			INDArray inputModGateActivations = miniBatchData.mmul(wg)
					.addi(prevOutputActivations.mmul(wG))
					.addi(prevMemCellActivations.mmul(Nd4j.diag(wGG)))
					.addiRowVector(bg);
			NDArrayIndex[] gIndexes = (miniBatchSize == 1 ? new NDArrayIndex[]{interval(3*hiddenLayerSize,4*hiddenLayerSize)} :
					new NDArrayIndex[]{NDArrayIndex.all(),interval(3*hiddenLayerSize,4*hiddenLayerSize)});
			ifogZ.slice(t,2).put(gIndexes, inputModGateActivations);
			ifogA.slice(t,2).put(gIndexes,
					Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", inputModGateActivations)));

			//Memory cell activations: (s_t then tanh(s_t))
			INDArray currentMemoryCellActivations = forgetGateActivations.mul(prevMemCellActivations)
					.addi(inputModGateActivations.mul(inputActivations));
			currentMemoryCellActivations = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(), currentMemoryCellActivations));

			INDArray outputGateActivations = miniBatchData.mmul(wo)
					.addi(prevOutputActivations.mmul(wO))
					.addi(currentMemoryCellActivations.mmul(Nd4j.diag(wOO)))
					.addiRowVector(bo);
			NDArrayIndex[] oIndexes = (miniBatchSize == 1 ? new NDArrayIndex[]{interval(2*hiddenLayerSize,3*hiddenLayerSize)} :
					new NDArrayIndex[]{NDArrayIndex.all(),interval(2*hiddenLayerSize,3*hiddenLayerSize)});
			ifogZ.slice(t,2).put(oIndexes, outputGateActivations);
			ifogA.slice(t,2).put(oIndexes,Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", outputGateActivations)));

			//LSTM unit outputs:
			INDArray currHiddenUnitActivations = outputGateActivations.mul(currentMemoryCellActivations);	//Expected shape: [m,hiddenLayerSize]

			outputActivations.slice(t,2).assign(currHiddenUnitActivations);
			memCellActivations.slice(t,2).assign(currentMemoryCellActivations);
		}
		// TODO Verify - this is from commit on 2d case handling for activation in GravesLSTM. Probably needs to be address with new take on epsilon
		if (is2dInput) {
			int[] shape = outputActivations.shape();
			outputActivations = outputActivations.reshape(shape[0], shape[1]);
		}

		return new INDArray[]{outputActivations,memCellActivations,ifogZ,ifogA};
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
}
