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
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**LSTM layer implementation.
 * Based on Graves: Supervised Sequence Labelling with Recurrent Neural Networks
 * http://www.cs.toronto.edu/~graves/phd.pdf
 * See also for full/vectorized equations (and a comparison to other LSTM variants):
 * Greff et al. 2015, "LSTM: A Search Space Odyssey", pg11. This is the "vanilla" variant in said paper
 * http://arxiv.org/pdf/1503.04069.pdf
 *
 * @author Alex Black
 */
public class GravesLSTM extends BaseLayer<org.deeplearning4j.nn.conf.layers.GravesLSTM> {

	public GravesLSTM(NeuralNetConfiguration conf) {
		super(conf);
	}

	public GravesLSTM(NeuralNetConfiguration conf, INDArray input) {
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
		//First: Do forward pass to get gate activations etc.
		INDArray[] activations = activateHelper(true);	//Order: {outputActivations,memCellActivations,ifogZs,ifogAs}
		INDArray outputActivations = activations[0];
		INDArray memCellState = activations[1];
		INDArray ifogZs = activations[2];
		INDArray ifogAs = activations[3];
		
		INDArray inputWeights = getParam(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);
		INDArray recurrentWeights = getParam(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY);	//Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]

		//Expect errors to have shape: [miniBatchSize,n^(L+1),timeSeriesLength]
		int hiddenLayerSize = recurrentWeights.size(0);	//i.e., n^L
		int prevLayerSize = inputWeights.size(0);	//n^(L-1)
		int miniBatchSize = epsilon.size(0);
		boolean is2dInput = epsilon.rank() < 3; //Edge case: T=1 may have shape [miniBatchSize,n^(L+1)], equiv. to [miniBatchSize,n^(L+1),1]
		int timeSeriesLength = (is2dInput? 1: epsilon.size(2));
		
		INDArray wi = inputWeights.get(NDArrayIndex.all(),interval(0,hiddenLayerSize));	//i.e., want rows 0..nIn, columns 0..hiddenLayerSize
		INDArray wI = recurrentWeights.get(NDArrayIndex.all(),interval(0,hiddenLayerSize));
		INDArray wf = inputWeights.get(NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize));
		INDArray wF = recurrentWeights.get(NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize));
		INDArray wFF = recurrentWeights.get(NDArrayIndex.all(),interval(4*hiddenLayerSize,4 * hiddenLayerSize + 1));
		INDArray wo = inputWeights.get(NDArrayIndex.all(),interval(2 * hiddenLayerSize,3 * hiddenLayerSize));
		INDArray wO = recurrentWeights.get(NDArrayIndex.all(),interval(2 * hiddenLayerSize,3 * hiddenLayerSize));
		INDArray wOO = recurrentWeights.get(NDArrayIndex.all(),interval(4 * hiddenLayerSize + 1,4 * hiddenLayerSize + 2));
		INDArray wg = inputWeights.get(NDArrayIndex.all(),interval(3 * hiddenLayerSize,4*hiddenLayerSize));
		INDArray wG = recurrentWeights.get(NDArrayIndex.all(),interval(3*hiddenLayerSize,4 * hiddenLayerSize));
		INDArray wGG = recurrentWeights.get(NDArrayIndex.all(),interval(4*hiddenLayerSize + 2,4 * hiddenLayerSize + 3));

		//Parameter gradients, summed across time. bias gradients, input weight gradients, recurrent weight gradients
		INDArray bGradients = Nd4j.zeros(1,4*hiddenLayerSize);
		INDArray iwGradients = Nd4j.zeros(prevLayerSize,4*hiddenLayerSize);
		INDArray rwGradients = Nd4j.zeros(hiddenLayerSize,4*hiddenLayerSize+3);	//Order: {I,F,O,G,FF,OO,GG}

		INDArray epsilonNext = Nd4j.zeros(miniBatchSize,prevLayerSize,timeSeriesLength);	//i.e., what would be W^L*(delta^L)^T. Shape: [m,n^(L-1),T]
		
		/*Placeholder. To be replaced by masking array for used for variable length time series
		 *Idea: M[i,j] = 1 if data is present for time j in example i in mini-batch.
		 *M[i,j] = 0 otherwise
		 *Then do a column multiply to set appropriate deltas to 0 if data is beyond end of time series
		 *for the corresponding example
		 */
//		INDArray timeSeriesMaskArray = Nd4j.ones(miniBatchSize,timeSeriesLength);	//For now: assume that all data in mini-batch is of length 'timeSeriesLength'

		INDArray nablaCellStateNext = Nd4j.zeros(miniBatchSize,hiddenLayerSize);
		INDArray deltaiNext = Nd4j.zeros(miniBatchSize,hiddenLayerSize);
		INDArray deltafNext = Nd4j.zeros(miniBatchSize,hiddenLayerSize);
		INDArray deltaoNext = Nd4j.zeros(miniBatchSize,hiddenLayerSize);
		INDArray deltagNext = Nd4j.zeros(miniBatchSize,hiddenLayerSize);
		
		for( int t=timeSeriesLength-1; t>=0; t-- ){
			INDArray prevMemCellState = (t==0 ? Nd4j.zeros(miniBatchSize, hiddenLayerSize) : memCellState.tensorAlongDimension(t-1,1,0) );	//Shape: [m, n^L]
			INDArray prevHiddenUnitActivation = (t==0 ? Nd4j.zeros(miniBatchSize, hiddenLayerSize) : outputActivations.tensorAlongDimension(t-1,1,0) );	//Shape: [m, n^L]; i.e., layer output at prev. time step.
			INDArray currMemCellState = (is2dInput ? memCellState : memCellState.tensorAlongDimension(t,1,0) );

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
			INDArray epsilonSlice = (is2dInput ? epsilon : epsilon.tensorAlongDimension(t,1,0));		//(w^{L+1}*(delta^{(L+1)t})^T)^T or equiv.
			INDArray nablaOut = epsilonSlice.dup()
					.addi(deltaiNext.mmul(wI.transpose()))
					.addi(deltafNext.mmul(wF.transpose()))
					.addi(deltaoNext.mmul(wO.transpose()))
					.addi(deltagNext.mmul(wG.transpose()));
			//Shape: [m,n^L]

			//Output gate deltas:
			INDArray sigmahOfS = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), currMemCellState.dup()));//	shape: [m,n^L]
			INDArray zo;
			if( is2dInput ) zo = ifogZs.get(NDArrayIndex.all(),interval(2*hiddenLayerSize,3*hiddenLayerSize));
			else zo = ifogZs.tensorAlongDimension(t,1,0).get(NDArrayIndex.all(),interval(2*hiddenLayerSize,3*hiddenLayerSize));
			INDArray sigmaoPrimeOfZo = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", zo.dup()).derivative());//			shape: [m,n^L]
			INDArray deltao = nablaOut.mul(sigmahOfS).muli(sigmaoPrimeOfZo); //Shape: [m,n^L]

			//Memory cell error:
			INDArray sigmahPrimeOfS = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), currMemCellState.dup()).derivative());//	shape: [m,n^L]
			INDArray ao;
			if( is2dInput ) ao = ifogAs.get(NDArrayIndex.all(),interval(2*hiddenLayerSize,3*hiddenLayerSize));
			else ao = ifogAs.tensorAlongDimension(t,1,0).get(NDArrayIndex.all(),interval(2*hiddenLayerSize,3*hiddenLayerSize));
			INDArray nextForgetGateAs = (t==timeSeriesLength-1 ? Nd4j.zeros(miniBatchSize,hiddenLayerSize) :	//t==0 should also cover is2dInput case
					ifogAs.tensorAlongDimension(t+1,1,0).get(NDArrayIndex.all(),interval(hiddenLayerSize,2*hiddenLayerSize)) );
			INDArray nablaCellState = nablaOut.mul(ao).muli(sigmahPrimeOfS)
					.addi(nextForgetGateAs.mul(nablaCellStateNext))
					.addi(deltafNext.mulRowVector(wFF.transpose()))
					.addi(deltao.mulRowVector(wOO.transpose()))
					.addi(deltagNext.mulRowVector(wGG.transpose()));
			nablaCellStateNext = nablaCellState;	//Store for use in next iteration

			//Forget gate delta:
			INDArray zf = (is2dInput ? ifogZs.get(NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize))	//z_f^{Lt}	shape: [m,n^L] 
					: ifogZs.tensorAlongDimension(t,1,0).get(NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize)) );
			INDArray deltaf = nablaCellState.mul(prevMemCellState)
					.muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", zf.dup()).derivative()));
			//Shape: [m,n^L]

			//Input modulation gate delta:
			INDArray zg = (is2dInput ? ifogZs.get(NDArrayIndex.all(),interval(3*hiddenLayerSize,4 * hiddenLayerSize))	//z_g^{Lt}	shape: [m,n^L] 
					: ifogZs.tensorAlongDimension(t,1,0).get(NDArrayIndex.all(),interval(3*hiddenLayerSize,4 * hiddenLayerSize)) );
			INDArray ai = (is2dInput ? ifogAs.get(NDArrayIndex.all(),interval(0,hiddenLayerSize)) 	//a_i^{Lt}	shape: [m,n^L]
					: ifogAs.tensorAlongDimension(t,1,0).get(NDArrayIndex.all(),interval(0,hiddenLayerSize)) );
			INDArray deltag = nablaCellState.mul(ai)
					.muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", zg.dup()).derivative()));
			//Shape: [m,n^L]

			//Network input delta:
			INDArray zi = (is2dInput ? ifogZs.get(NDArrayIndex.all(),interval(0,hiddenLayerSize))	//z_i^{Lt}	shape: [m,n^L]
					: ifogZs.tensorAlongDimension(t,1,0).get(NDArrayIndex.all(),interval(0,hiddenLayerSize)) );
			INDArray ag = (is2dInput ? ifogAs.get(NDArrayIndex.all(),interval(3*hiddenLayerSize,4 * hiddenLayerSize)) //a_g^{Lt}	shape: [m,n^L]
					: ifogAs.tensorAlongDimension(t,1,0).get(NDArrayIndex.all(),interval(3*hiddenLayerSize,4 * hiddenLayerSize)) );	
			INDArray deltai = nablaCellState.mul(ag)
					.muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), zi.dup()).derivative()));
			//Shape: [m,n^L]

			INDArray prevLayerActivationSlice = (is2dInput ? input : input.tensorAlongDimension(t,1,0));
			iwGradients.get(new INDArrayIndex[]{NDArrayIndex.all(),interval(0,hiddenLayerSize)}).addi(deltai.transpose().mmul(prevLayerActivationSlice).transpose());
			iwGradients.get(new INDArrayIndex[]{NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize)}).addi(deltaf.transpose().mmul(prevLayerActivationSlice).transpose());
			iwGradients.get(new INDArrayIndex[]{NDArrayIndex.all(),interval(2 * hiddenLayerSize,3 * hiddenLayerSize)}).addi(deltao.transpose().mmul(prevLayerActivationSlice).transpose());
			iwGradients.get(new INDArrayIndex[]{NDArrayIndex.all(),interval(3 * hiddenLayerSize,4 * hiddenLayerSize)}).addi(deltag.transpose().mmul(prevLayerActivationSlice).transpose());

			if( t > 0 ){
				//Minor optimization. If t==0, then prevHiddenUnitActivation==zeros(n^L,n^L), so dL/dW for recurrent weights will end up as 0 anyway. (They are initialized as 0)
				rwGradients.get(new INDArrayIndex[]{NDArrayIndex.all(),interval(0,hiddenLayerSize)}).addi(deltai.transpose().mmul(prevHiddenUnitActivation).transpose());	//dL/dw_{Ixy} = delta_{ix} * a_{iy}^{L(t-1)}
				rwGradients.get(new INDArrayIndex[]{NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize)}).addi(deltaf.transpose().mmul(prevHiddenUnitActivation).transpose());	//dL/dw_{Fxy} = delta_{fx} * a_{iy}^{L(t-1)}
				rwGradients.get(new INDArrayIndex[]{NDArrayIndex.all(),interval(2 * hiddenLayerSize,3 * hiddenLayerSize)}).addi(deltao.transpose().mmul(prevHiddenUnitActivation).transpose());	//dL/dw_{O}
				rwGradients.get(new INDArrayIndex[]{NDArrayIndex.all(),interval(3 * hiddenLayerSize,4 * hiddenLayerSize)}).addi(deltag.transpose().mmul(prevHiddenUnitActivation).transpose());	//dL/dw_{G}

				//Expected shape: [n^L,1]. sum(0) is sum over examples in mini-batch.
				INDArray dLdwFF = deltaf.mul(prevMemCellState).sum(0).transpose();	//mul not mmul because these weights are from unit j->j only (whereas other recurrent weights are i->j for all i,j)
				rwGradients.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(4*hiddenLayerSize)}).addi(dLdwFF);	//dL/dw_{FF}
				INDArray dLdwGG = deltag.mul(prevMemCellState).sum(0).transpose();
				rwGradients.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(4*hiddenLayerSize + 2)}).addi(dLdwGG);	//dL/dw_{GG}
			}
			INDArray dLdwOO = deltao.mul(currMemCellState).sum(0).transpose();	//Expected shape: [n^L,1]. sum(0) is sum over examples in mini-batch.
			rwGradients.get(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(4*hiddenLayerSize + 1)}).addi(dLdwOO);	//dL/dw_{OOxy}

			bGradients.get(new INDArrayIndex[]{NDArrayIndex.point(0),interval(0,hiddenLayerSize)}).addi(deltai.sum(0));
			bGradients.get(new INDArrayIndex[]{NDArrayIndex.point(0),interval(hiddenLayerSize,2 * hiddenLayerSize)}).addi(deltaf.sum(0));
			bGradients.get(new INDArrayIndex[]{NDArrayIndex.point(0),interval(2*hiddenLayerSize,3 * hiddenLayerSize)}).addi(deltao.sum(0));
			bGradients.get(new INDArrayIndex[]{NDArrayIndex.point(0),interval(3*hiddenLayerSize,4 * hiddenLayerSize)}).addi(deltag.sum(0));
			
			//Calculate epsilonNext - i.e., equiv. to what would be (w^L*(d^(Lt))^T)^T in a normal network
			//But here, need to add 4 weights * deltas for the IFOG gates
			INDArray epsilonNextSlice = wi.mmul(deltai.transpose()).transpose()
					.addi(wf.mmul(deltaf.transpose()).transpose())
					.addi(wo.mmul(deltao.transpose()).transpose())
					.addi(wg.mmul(deltag.transpose()).transpose());
			epsilonNext.tensorAlongDimension(t,1,0).assign(epsilonNextSlice);
			
			deltaiNext = deltai;
			deltafNext = deltaf;
			deltaoNext = deltao;
			deltagNext = deltag;
		}

		//Weight/bias gradients: sum across time dimension. But leave mini-batch dimension for time (in keeping with what BaseLayer.update() expects.
		Gradient retGradient = new DefaultGradient();
		retGradient.gradientForVariable().put(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY,iwGradients);
		retGradient.gradientForVariable().put(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY,rwGradients);
		retGradient.gradientForVariable().put(GravesLSTMParamInitializer.BIAS_KEY, bGradients);

		return new Pair<>(retGradient,epsilonNext);
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

	/**Returns 4 INDArrays: [outputActivations, memCellState, ifogZs, ifogAs] in that order.
	 * Need all 4 to do backward pass, but only care about the first one for forward pass.
	 */
	private INDArray[] activateHelper(boolean training){
		//Mini-batch data format: for mini-batch size m, nIn inputs, and T time series length
		//Data has shape [m,nIn,T]. Layer activations/output has shape [m,nHiddenUnits,T]

		INDArray recurrentWeights = getParam(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY);	//Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
		INDArray inputWeights = getParam(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);			//Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
		INDArray biases = getParam(GravesLSTMParamInitializer.BIAS_KEY); //by row: IFOG			//Shape: [4,hiddenLayerSize]; order: [bi,bf,bo,bg]^T

		boolean is2dInput = input.rank() < 3;		//Edge case of T=1, may have shape [m,nIn], equiv. to [m,nIn,1]
		int timeSeriesLength = (is2dInput ? 1 : input.size(2));
		int hiddenLayerSize = recurrentWeights.size(0);
		int miniBatchSize = input.size(0);
		int nIn = inputWeights.size(0);		//Size of previous layer (or input)

		//Apply dropconnect to input (not recurrent) weights only:
		if(conf.isUseDropConnect() && training) {
			if (conf.getLayer().getDropOut() > 0) {
				inputWeights = Dropout.applyDropConnect(this,GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);
			}
		}

		//Extract weights and biases:
		INDArray wi = inputWeights.get(NDArrayIndex.all(),interval(0,hiddenLayerSize));	//i.e., want rows 0..nIn, columns 0..hiddenLayerSize
		INDArray wI = recurrentWeights.get(NDArrayIndex.all(),interval(0,hiddenLayerSize));
		INDArray bi = biases.get(NDArrayIndex.point(0),interval(0,hiddenLayerSize));

		INDArray wf = inputWeights.get(NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize));
		INDArray wF = recurrentWeights.get(NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize)); //previous
		INDArray wFF = recurrentWeights.get(NDArrayIndex.all(),interval(4*hiddenLayerSize,4 * hiddenLayerSize + 1)); //current
		INDArray bf = biases.get(NDArrayIndex.point(0),interval(hiddenLayerSize,2*hiddenLayerSize));

		INDArray wo = inputWeights.get(NDArrayIndex.all(),interval(2 * hiddenLayerSize,3 * hiddenLayerSize));
		INDArray wO = recurrentWeights.get(NDArrayIndex.all(),interval(2 * hiddenLayerSize,3 * hiddenLayerSize)); //previous
		INDArray wOO = recurrentWeights.get(NDArrayIndex.all(),interval(4 * hiddenLayerSize + 1,4 * hiddenLayerSize + 2)); //current
		INDArray bo = biases.get(NDArrayIndex.point(0),interval(2*hiddenLayerSize,3 * hiddenLayerSize));

		INDArray wg = inputWeights.get(NDArrayIndex.all(),interval(3 * hiddenLayerSize,4*hiddenLayerSize));
		INDArray wG = recurrentWeights.get(NDArrayIndex.all(),interval(3*hiddenLayerSize,4 * hiddenLayerSize)); //previous
		INDArray wGG = recurrentWeights.get(NDArrayIndex.all(),interval(4*hiddenLayerSize + 2,4 * hiddenLayerSize + 3)); //previous
		INDArray bg = biases.get(NDArrayIndex.point(0),interval(3*hiddenLayerSize,4 * hiddenLayerSize));

		//Allocate arrays for activations:
		INDArray outputActivations = Nd4j.zeros(new int[]{miniBatchSize,hiddenLayerSize,timeSeriesLength});
		INDArray ifogZ = Nd4j.zeros(new int[]{miniBatchSize,4 * hiddenLayerSize,timeSeriesLength});
		INDArray ifogA = Nd4j.zeros(new int[]{miniBatchSize,4 * hiddenLayerSize,timeSeriesLength});
		INDArray memCellState = Nd4j.zeros(new int[]{miniBatchSize,hiddenLayerSize,timeSeriesLength});


		for( int t = 0; t < timeSeriesLength; t++ ){
			INDArray miniBatchData = (is2dInput ? input : input.tensorAlongDimension(t,1,0));	//[Expected shape: [m,nIn]. Also deals with edge case of T=1, with 'time series' data of shape [m,nIn], equiv. to [m,nIn,1]
			INDArray prevOutputActivations = (t==0 ? Nd4j.zeros(new int[]{miniBatchSize,hiddenLayerSize}) : outputActivations.tensorAlongDimension(t-1,1,0));	//Shape: [m,nL]
			INDArray prevMemCellState = (t==0 ? Nd4j.zeros(new int[]{miniBatchSize,hiddenLayerSize}) : memCellState.tensorAlongDimension(t-1,1,0));	//Shape: [m,nL]

			//Calculate activations for: network input + forget, output, input modulation gates.
			INDArray inputActivations = miniBatchData.mmul(wi)
					.addi(prevOutputActivations.mmul(wI))
					.addiRowVector(bi);
			INDArrayIndex[] iIndexes = new INDArrayIndex[]{NDArrayIndex.all(),interval(0,hiddenLayerSize)};
			ifogZ.tensorAlongDimension(t,1,0).put(iIndexes, inputActivations);
			Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), inputActivations));
			ifogA.tensorAlongDimension(t,1,0).put(iIndexes, inputActivations);

			INDArray forgetGateActivations = miniBatchData.mmul(wf)
					.addi(prevOutputActivations.mmul(wF))
					.addi(prevMemCellState.mulRowVector(wFF.transpose()))
					.addiRowVector(bf);
			INDArrayIndex[] fIndexes = new INDArrayIndex[]{NDArrayIndex.all(),interval(hiddenLayerSize,2*hiddenLayerSize)};
			ifogZ.tensorAlongDimension(t,1,0).put(fIndexes, forgetGateActivations);
			Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", forgetGateActivations));
			ifogA.tensorAlongDimension(t,1,0).put(fIndexes, forgetGateActivations);

			INDArray inputModGateActivations = miniBatchData.mmul(wg)
					.addi(prevOutputActivations.mmul(wG))
					.addi(prevMemCellState.mulRowVector(wGG.transpose()))
					.addiRowVector(bg);
			INDArrayIndex[] gIndexes = new INDArrayIndex[]{NDArrayIndex.all(),interval(3*hiddenLayerSize,4*hiddenLayerSize)};
			ifogZ.tensorAlongDimension(t,1,0).put(gIndexes, inputModGateActivations);
			Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", inputModGateActivations));
			ifogA.tensorAlongDimension(t,1,0).put(gIndexes, inputModGateActivations);

			//Memory cell state
			INDArray currentMemoryCellState = forgetGateActivations.mul(prevMemCellState)
					.addi(inputModGateActivations.mul(inputActivations));

			INDArray outputGateActivations = miniBatchData.mmul(wo)
					.addi(prevOutputActivations.mmul(wO))
					.addi(currentMemoryCellState.mulRowVector(wOO.transpose()))
					.addiRowVector(bo);
			INDArrayIndex[] oIndexes = new INDArrayIndex[]{NDArrayIndex.all(),interval(2*hiddenLayerSize,3*hiddenLayerSize)};
			ifogZ.tensorAlongDimension(t,1,0).put(oIndexes, outputGateActivations);
			Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", outputGateActivations));
			ifogA.tensorAlongDimension(t,1,0).put(oIndexes, outputGateActivations);

			//LSTM unit outputs:
			INDArray currMemoryCellActivation = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), currentMemoryCellState.dup()));
			INDArray currHiddenUnitActivations = outputGateActivations.mul(currMemoryCellActivation);	//Expected shape: [m,hiddenLayerSize]

			outputActivations.tensorAlongDimension(t,1,0).assign(currHiddenUnitActivations);
			memCellState.tensorAlongDimension(t,1,0).assign(currentMemoryCellState);
		}

		return new INDArray[]{outputActivations,memCellState,ifogZ,ifogA};
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
    	double l2 = Transforms.pow(getParam(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY), 2).sum(Integer.MAX_VALUE).getDouble(0)
    			+ Transforms.pow(getParam(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY), 2).sum(Integer.MAX_VALUE).getDouble(0);
    	return 0.5 * conf.getL2() * l2;
    }

    @Override
    public double calcL1() {
    	if(!conf.isUseRegularization() || conf.getL1() <= 0.0 ) return 0.0;
        double l1 = Transforms.abs(getParam(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY)).sum(Integer.MAX_VALUE).getDouble(0)
        		+ Transforms.abs(getParam(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY)).sum(Integer.MAX_VALUE).getDouble(0);
        return conf.getL1() * l1;
    }
}
