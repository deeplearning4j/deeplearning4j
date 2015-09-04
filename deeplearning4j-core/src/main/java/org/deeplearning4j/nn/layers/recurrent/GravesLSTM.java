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
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
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
public class GravesLSTM extends BaseRecurrentLayer<org.deeplearning4j.nn.conf.layers.GravesLSTM> {
	public static final String STATE_KEY_PREV_ACTIVATION = "prevAct";
	public static final String STATE_KEY_PREV_MEMCELL = "prevMem";

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
		return backpropGradientHelper(epsilon,false);
	}

	@Override
	public Pair<Gradient, INDArray> tbpttBackpropGradient(INDArray epsilon){
		return backpropGradientHelper(epsilon,true);
	}

	private Pair<Gradient,INDArray> backpropGradientHelper(INDArray epsilon, boolean truncatedBPTT){
		//First: Do forward pass to get gate activations, zs etc.
		FwdPassReturn fwdPass;
		if(truncatedBPTT){
			fwdPass = activateHelper(true,stateMap.get(STATE_KEY_PREV_ACTIVATION),stateMap.get(STATE_KEY_PREV_MEMCELL),true);
			//Store last time step of output activations and memory cell state in tBpttStateMap
			tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct);
			tBpttStateMap.put(STATE_KEY_PREV_MEMCELL, fwdPass.lastMemCell);
		} else {
			fwdPass = activateHelper(true,null,null,true);
		}
		
		INDArray inputWeights = getParam(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);
		INDArray recurrentWeights = getParam(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY);	//Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]

		//Expect errors to have shape: [miniBatchSize,n^(L+1),timeSeriesLength]
		int hiddenLayerSize = recurrentWeights.size(0);	//i.e., n^L
		int prevLayerSize = inputWeights.size(0);	//n^(L-1)
		int miniBatchSize = epsilon.size(0);
		boolean is2dInput = epsilon.rank() < 3; //Edge case: T=1 may have shape [miniBatchSize,n^(L+1)], equiv. to [miniBatchSize,n^(L+1),1]
		int timeSeriesLength = (is2dInput? 1: epsilon.size(2));

		for( int i=0; i<fwdPass.paramsZeroOffset.length; i++ ) fwdPass.paramsZeroOffset[i] = Shape.toOffsetZero(fwdPass.paramsZeroOffset[i].transpose());

		INDArray wiTranspose = fwdPass.paramsZeroOffset[0];
		INDArray wITranspose = fwdPass.paramsZeroOffset[1];
		INDArray wfTranspose = fwdPass.paramsZeroOffset[2];
		INDArray wFTranspose = fwdPass.paramsZeroOffset[3];
		INDArray wFFTranspose = fwdPass.paramsZeroOffset[4];
		INDArray woTranspose = fwdPass.paramsZeroOffset[5];
		INDArray wOTranspose = fwdPass.paramsZeroOffset[6];
		INDArray wOOTranspose = fwdPass.paramsZeroOffset[7];
		INDArray wgTranspose = fwdPass.paramsZeroOffset[8];
		INDArray wGTranspose = fwdPass.paramsZeroOffset[9];
		INDArray wGGTranspose = fwdPass.paramsZeroOffset[10];

		//Parameter gradients, summed across time. bias gradients, input weight gradients, recurrent weight gradients
		INDArray[] bGradients = new INDArray[4];
		INDArray[] iwGradients = new INDArray[4];
		INDArray[] rwGradients = new INDArray[7];	//Order: {I,F,O,G,FF,OO,GG}
		for( int i=0; i<4; i++ ){
			bGradients[i] = Nd4j.zeros(1,hiddenLayerSize);
			iwGradients[i] = Nd4j.zeros(prevLayerSize,hiddenLayerSize);
			rwGradients[i] = Nd4j.zeros(hiddenLayerSize,hiddenLayerSize);
		}
		for( int i=0; i<3; i++ ) rwGradients[i+4] = Nd4j.zeros(1,hiddenLayerSize);

		INDArray epsilonNext = Nd4j.zeros(miniBatchSize,prevLayerSize,timeSeriesLength);	//i.e., what would be W^L*(delta^L)^T. Shape: [m,n^(L-1),T]
		
		/*Placeholder. To be replaced by masking array for used for variable length time series
		 *Idea: M[i,j] = 1 if data is present for time j in example i in mini-batch.
		 *M[i,j] = 0 otherwise
		 *Then do a column multiply to set appropriate deltas to 0 if data is beyond end of time series
		 *for the corresponding example
		 */
//		INDArray timeSeriesMaskArray = Nd4j.ones(miniBatchSize,timeSeriesLength);	//For now: assume that all data in mini-batch is of length 'timeSeriesLength'

		INDArray nablaCellStateNext = Nd4j.zeros(miniBatchSize,hiddenLayerSize);
		INDArray deltaiNext = null;
		INDArray deltafNext = Nd4j.zeros(miniBatchSize,hiddenLayerSize);
		INDArray deltaoNext = null;
		INDArray deltagNext = Nd4j.zeros(miniBatchSize,hiddenLayerSize);
		
		for( int t=timeSeriesLength-1; t>=0; t-- ){
			INDArray prevMemCellState = (t==0 ? Nd4j.zeros(miniBatchSize, hiddenLayerSize) : fwdPass.memCellState[t-1]);
			INDArray prevHiddenUnitActivation = (t==0 ? null : fwdPass.fwdPassOutputAsArrays[t-1] );
			INDArray currMemCellState = fwdPass.memCellState[t];

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
			INDArray nablaOut = epsilonSlice; //Shape: [m,n^L]
			if(t!=timeSeriesLength-1){
				//if t == timeSeriesLength-1 then deltaiNext etc are zeros
				nablaOut = nablaOut.dup();
				nablaOut.addi(deltaiNext.mmul(wITranspose))
					.addi(deltafNext.mmul(wFTranspose))
					.addi(deltaoNext.mmul(wOTranspose))
					.addi(deltagNext.mmul(wGTranspose));
			}

			//Output gate deltas:
			INDArray sigmahOfS = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), currMemCellState.dup()));//	shape: [m,n^L]
			INDArray zo = fwdPass.oz[t];
			INDArray sigmaoPrimeOfZo = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", zo).derivative()); //shape: [m,n^L]
			//Normally would use zo.dup() in above line, but won't be using zo again (for this time step). Ditto for zf, zg, zi
			INDArray deltao = nablaOut.mul(sigmahOfS).muli(sigmaoPrimeOfZo); //Shape: [m,n^L]

			//Memory cell error:
			INDArray sigmahPrimeOfS = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), currMemCellState.dup()).derivative());//	shape: [m,n^L]
			INDArray ao = fwdPass.oa[t];
			INDArray nextForgetGateAs = (t==timeSeriesLength-1 ? Nd4j.zeros(miniBatchSize,hiddenLayerSize) : fwdPass.fa[t+1]);
			INDArray nablaCellState = nablaOut.mul(ao).muli(sigmahPrimeOfS)
					.addi(nextForgetGateAs.mul(nablaCellStateNext))
					.addi(deltafNext.mulRowVector(wFFTranspose))
					.addi(deltao.mulRowVector(wOOTranspose))
					.addi(deltagNext.mulRowVector(wGGTranspose));
			nablaCellStateNext = nablaCellState;	//Store for use in next iteration

			//Forget gate delta:
			INDArray zf = fwdPass.fz[t];
			INDArray deltaf = nablaCellState.mul(prevMemCellState)
					.muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", zf).derivative()));
			//Shape: [m,n^L]

			//Input modulation gate delta:
			INDArray zg = fwdPass.gz[t];
			INDArray ai = fwdPass.ia[t];
			INDArray deltag = nablaCellState.mul(ai)
					.muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", zg).derivative()));
			//Shape: [m,n^L]

			//Network input delta:
			INDArray zi = fwdPass.iz[t];
			INDArray ag = fwdPass.ga[t];
			INDArray deltai = nablaCellState.mul(ag)
					.muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), zi).derivative()));
			//Shape: [m,n^L]

			INDArray prevLayerActivationSliceTransposed = Shape.toOffsetZero(is2dInput ? input.transpose() : input.tensorAlongDimension(t,1,0).transpose());
			iwGradients[0].addi(prevLayerActivationSliceTransposed.mmul(deltai));
			iwGradients[1].addi(prevLayerActivationSliceTransposed.mmul(deltaf));
			iwGradients[2].addi(prevLayerActivationSliceTransposed.mmul(deltao));
			iwGradients[3].addi(prevLayerActivationSliceTransposed.mmul(deltag));

			if( t > 0 ){
				//Minor optimization. If t==0, then prevHiddenUnitActivation==zeros(n^L,n^L), so dL/dW for recurrent weights will end up as 0 anyway. (They are initialized as 0)
				INDArray prevActTranspose = Shape.toOffsetZero(prevHiddenUnitActivation.transpose());
				rwGradients[0].addi(prevActTranspose.mmul(deltai));
				rwGradients[1].addi(prevActTranspose.mmul(deltaf));
				rwGradients[2].addi(prevActTranspose.mmul(deltao));
				rwGradients[3].addi(prevActTranspose.mmul(deltag));

				//Shape: [1,n^L]. sum(0) is sum over examples in mini-batch.
				INDArray dLdwFF = deltaf.mul(prevMemCellState).sum(0);	//mul not mmul because these weights are from unit j->j only (whereas other recurrent weights are i->j for all i,j)
				rwGradients[4].addi(dLdwFF);	//dL/dw_{FF}
				INDArray dLdwGG = deltag.mul(prevMemCellState).sum(0);
				rwGradients[6].addi(dLdwGG);
			}
			INDArray dLdwOO = deltao.mul(currMemCellState).sum(0);	//Expected shape: [n^L,1]. sum(0) is sum over examples in mini-batch.
			rwGradients[5].addi(dLdwOO);	//dL/dw_{OOxy}

			bGradients[0].addi(deltai.sum(0));
			bGradients[1].addi(deltaf.sum(0));
			bGradients[2].addi(deltao.sum(0));
			bGradients[3].addi(deltag.sum(0));
			
			//Calculate epsilonNext - i.e., equiv. to what would be (w^L*(d^(Lt))^T)^T in a normal network
			//But here, need to add 4 weights * deltas for the IFOG gates
			INDArray epsilonNextSlice = deltai.mmul(wiTranspose)
					.addi(deltaf.mmul(wfTranspose))
					.addi(deltao.mmul(woTranspose))
					.addi(deltag.mmul(wgTranspose));
			epsilonNext.tensorAlongDimension(t,1,0).assign(epsilonNextSlice);
			
			deltaiNext = deltai;
			deltafNext = deltaf;
			deltaoNext = deltao;
			deltagNext = deltag;
		}

		//Weight/bias gradients
		INDArray iwGradientsOut = Nd4j.zeros(prevLayerSize,4*hiddenLayerSize);
		INDArray rwGradientsOut = Nd4j.zeros(hiddenLayerSize,4*hiddenLayerSize+3);	//Order: {I,F,O,G,FF,OO,GG}
		INDArray bGradientsOut = Nd4j.hstack(bGradients);
		iwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(),interval(0,hiddenLayerSize)},iwGradients[0]);
		iwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize)},iwGradients[1]);
		iwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(),interval(2 * hiddenLayerSize,3 * hiddenLayerSize)},iwGradients[2]);
		iwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(),interval(3 * hiddenLayerSize,4 * hiddenLayerSize)},iwGradients[3]);

		rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(),interval(0,hiddenLayerSize)},rwGradients[0]);
		rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(),interval(hiddenLayerSize,2 * hiddenLayerSize)},rwGradients[1]);
		rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(),interval(2 * hiddenLayerSize,3 * hiddenLayerSize)},rwGradients[2]);
		rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(),interval(3 * hiddenLayerSize,4 * hiddenLayerSize)},rwGradients[3]);
		rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(4*hiddenLayerSize)},rwGradients[4].transpose());
		rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(4*hiddenLayerSize + 1)},rwGradients[5].transpose());
		rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.point(4*hiddenLayerSize + 2)},rwGradients[6].transpose());

		Gradient retGradient = new DefaultGradient();
		retGradient.gradientForVariable().put(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY,iwGradientsOut);
		retGradient.gradientForVariable().put(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY,rwGradientsOut);
		retGradient.gradientForVariable().put(GravesLSTMParamInitializer.BIAS_KEY, bGradientsOut);

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
		return activateHelper(training,null,null,false).fwdPassOutput;
	}

	@Override
	public INDArray activate(INDArray input){
		setInput(input);
		return activateHelper(true,null,null,false).fwdPassOutput;
	}

	@Override
	public INDArray activate(boolean training){
		return activateHelper(training,null,null,false).fwdPassOutput;
	}

	@Override
	public INDArray activate(){
		return activateHelper(false,null,null,false).fwdPassOutput;
	}

	/**Returns FwdPassReturn object with activations/INDArrays. Allows activateHelper to be used for forward pass, backward pass
	 * and rnnTimeStep whilst being reasonably efficient for all
	 */
	private FwdPassReturn activateHelper(boolean training, INDArray prevOutputActivations, INDArray prevMemCellState, boolean forBackprop){
		//Mini-batch data format: for mini-batch size m, nIn inputs, and T time series length
		//Data has shape [m,nIn,T]. Layer activations/output has shape [m,nHiddenUnits,T]

		INDArray recurrentWeights = getParam(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY);	//Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
		INDArray inputWeights = getParam(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);			//Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
		INDArray biases = getParam(GravesLSTMParamInitializer.BIAS_KEY); //by row: IFOG			//Shape: [4,hiddenLayerSize]; order: [bi,bf,bo,bg]^T

		boolean is2dInput = input.rank() < 3;		//Edge case of T=1, may have shape [m,nIn], equiv. to [m,nIn,1]
		int timeSeriesLength = (is2dInput ? 1 : input.size(2));
		int hiddenLayerSize = recurrentWeights.size(0);
		int miniBatchSize = input.size(0);

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
		
		if(timeSeriesLength>1 || forBackprop){
			wi = Shape.toOffsetZero(wi);
			wI = Shape.toOffsetZero(wI);
			wf = Shape.toOffsetZero(wf);
			wF = Shape.toOffsetZero(wF);
			wFF = Shape.toOffsetZero(wFF);
			wo = Shape.toOffsetZero(wo);
			wO = Shape.toOffsetZero(wO);
			wOO = Shape.toOffsetZero(wOO);
			wg = Shape.toOffsetZero(wg);
			wG = Shape.toOffsetZero(wG);
			wGG = Shape.toOffsetZero(wGG);
			bi = Shape.toOffsetZero(bi);
			bf = Shape.toOffsetZero(bf);
			bo = Shape.toOffsetZero(bo);
			bg = Shape.toOffsetZero(bg);
		}

		//Allocate arrays for activations:
		INDArray outputActivations = null; //Nd4j.zeros(new int[]{miniBatchSize,hiddenLayerSize,timeSeriesLength});

		FwdPassReturn toReturn = new FwdPassReturn();
		if(forBackprop){
			toReturn.paramsZeroOffset = new INDArray[]{wi,wI,wf,wF,wFF,wo,wO,wOO,wg,wG,wGG};
			toReturn.fwdPassOutputAsArrays = new INDArray[timeSeriesLength];
			toReturn.memCellState = new INDArray[timeSeriesLength];
			toReturn.iz = new INDArray[timeSeriesLength];
			toReturn.ia = new INDArray[timeSeriesLength];
			toReturn.fz = new INDArray[timeSeriesLength];
			toReturn.fa = new INDArray[timeSeriesLength];
			toReturn.oz = new INDArray[timeSeriesLength];
			toReturn.oa = new INDArray[timeSeriesLength];
			toReturn.gz = new INDArray[timeSeriesLength];
			toReturn.ga = new INDArray[timeSeriesLength];
		} else {
			outputActivations = Nd4j.zeros(new int[]{miniBatchSize,hiddenLayerSize,timeSeriesLength});
			toReturn.fwdPassOutput = outputActivations;
		}

		if(prevOutputActivations == null) prevOutputActivations = Nd4j.zeros(new int[]{miniBatchSize,hiddenLayerSize});
		if(prevMemCellState == null) prevMemCellState = Nd4j.zeros(new int[]{miniBatchSize,hiddenLayerSize});
		for( int t = 0; t < timeSeriesLength; t++ ){
			INDArray miniBatchData = (is2dInput ? input : input.tensorAlongDimension(t,1,0));	//[Expected shape: [m,nIn]. Also deals with edge case of T=1, with 'time series' data of shape [m,nIn], equiv. to [m,nIn,1]
			miniBatchData = Shape.toOffsetZero(miniBatchData);

			//Calculate activations for: network input + forget, output, input modulation gates.
			INDArray inputActivations = miniBatchData.mmul(wi)
					.addi(prevOutputActivations.mmul(wI))
					.addiRowVector(bi);
			if(forBackprop) toReturn.iz[t] = inputActivations.dup();
			Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), inputActivations));
			if(forBackprop) toReturn.ia[t] = inputActivations;

			INDArray forgetGateActivations = miniBatchData.mmul(wf)
					.addi(prevOutputActivations.mmul(wF))
					.addi(prevMemCellState.mulRowVector(wFF.transpose()))
					.addiRowVector(bf);
			if(forBackprop) toReturn.fz[t] = forgetGateActivations.dup();
			Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", forgetGateActivations));
			if(forBackprop) toReturn.fa[t] = forgetGateActivations;


			INDArray inputModGateActivations = miniBatchData.mmul(wg)
					.addi(prevOutputActivations.mmul(wG))
					.addi(prevMemCellState.mulRowVector(wGG.transpose()))
					.addiRowVector(bg);
			if(forBackprop) toReturn.gz[t] = inputModGateActivations.dup();
			Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", inputModGateActivations));
			if(forBackprop) toReturn.ga[t] = inputModGateActivations;

			//Memory cell state
			INDArray currentMemoryCellState = forgetGateActivations.mul(prevMemCellState)
					.addi(inputModGateActivations.mul(inputActivations));

			INDArray outputGateActivations = miniBatchData.mmul(wo)
					.addi(prevOutputActivations.mmul(wO))
					.addi(currentMemoryCellState.mulRowVector(wOO.transpose()))
					.addiRowVector(bo);
			if(forBackprop) toReturn.oz[t] = outputGateActivations.dup();
			Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", outputGateActivations));
			if(forBackprop) toReturn.oa[t] = outputGateActivations;

			//LSTM unit outputs:
			INDArray currMemoryCellActivation = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), currentMemoryCellState.dup()));
			INDArray currHiddenUnitActivations = currMemoryCellActivation.muli(outputGateActivations);	//Expected shape: [m,hiddenLayerSize]

			if(forBackprop){
				toReturn.fwdPassOutputAsArrays[t] = currHiddenUnitActivations;
				toReturn.memCellState[t] = currentMemoryCellState;
			} else {
				outputActivations.tensorAlongDimension(t,1,0).assign(currHiddenUnitActivations);
			}

			prevOutputActivations = currHiddenUnitActivations;
			prevMemCellState = currentMemoryCellState;

			toReturn.lastAct = currHiddenUnitActivations;
			toReturn.lastMemCell = currentMemoryCellState;
		}

		return toReturn;
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

	@Override
	public INDArray rnnTimeStep(INDArray input) {
		setInput(input);
		FwdPassReturn fwdPass = activateHelper(false,stateMap.get(STATE_KEY_PREV_ACTIVATION),stateMap.get(STATE_KEY_PREV_MEMCELL),false);
		INDArray outAct = fwdPass.fwdPassOutput;
		//Store last time step of output activations and memory cell state for later use:
		stateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct);
		stateMap.put(STATE_KEY_PREV_MEMCELL, fwdPass.lastMemCell);

		return outAct;
	}

	private static class FwdPassReturn {
		//First: needed by standard forward pass only
		private INDArray fwdPassOutput;
		//Arrays: Needed for backpropGradient only
		private INDArray[] paramsZeroOffset;	//{wi,wI,wf,wF,wFF,wo,wO,wOO,wg,wG,wGG}
		private INDArray[] fwdPassOutputAsArrays;
		private INDArray[] memCellState;
		private INDArray[] iz;
		private INDArray[] ia;
		private INDArray[] fz;
		private INDArray[] fa;
		private INDArray[] oz;
		private INDArray[] oa;
		private INDArray[] gz;
		private INDArray[] ga;
		//Last 2: needed for rnnTimeStep only
		private INDArray lastAct;
		private INDArray lastMemCell;
	}

	@Override
	public INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT) {
		setInput(input);
		FwdPassReturn fwdPass = activateHelper(training,stateMap.get(STATE_KEY_PREV_ACTIVATION),stateMap.get(STATE_KEY_PREV_MEMCELL),false);
		INDArray outAct = fwdPass.fwdPassOutput;
		if(storeLastForTBPTT){
			//Store last time step of output activations and memory cell state in tBpttStateMap
			tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct);
			tBpttStateMap.put(STATE_KEY_PREV_MEMCELL, fwdPass.lastMemCell);
		}

		return outAct;
	}
}
