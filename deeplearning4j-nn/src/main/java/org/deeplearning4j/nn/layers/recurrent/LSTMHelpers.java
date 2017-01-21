package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.TimesOneMinus;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.MulOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.Map;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 *
 * RNN tutorial: http://deeplearning4j.org/usingrnns.html
 * READ THIS FIRST if you want to understand what the heck is happening here.
 *
 * Shared code for the standard "forwards" LSTM RNN and the bidirectional LSTM RNN
 * This was extracted from GravesLSTM and refactored into static helper functions.  The general reasoning for this was
 * so we only have math in one place, instead of two.
 *
 * Based on Graves: Supervised Sequence Labelling with Recurrent Neural Networks
 * http://www.cs.toronto.edu/~graves/phd.pdf
 * See also for full/vectorized equations (and a comparison to other LSTM variants):
 * Greff et al. 2015, "LSTM: A Search Space Odyssey", pg11. This is the "vanilla" variant in said paper
 * http://arxiv.org/pdf/1503.04069.pdf
 *
 * Please note that truncated backpropagation through time (BPTT) will not work with the bidirectional layer as-is.
 * Additionally, variable length data sets will also not work with the bidirectional layer.
 *
 * @author Alex Black (LSTM implementation)
 * @author Benjamin Joseph (refactoring for bidirectional LSTM)
 */
public class LSTMHelpers {

//    public static final String SIGMOID = "sigmoid";

    private LSTMHelpers() {
    }

    /**
     * Returns FwdPassReturn object with activations/INDArrays. Allows activateHelper to be used for forward pass, backward pass
     * and rnnTimeStep whilst being reasonably efficient for all
     */
    static public FwdPassReturn activateHelper(final Layer layer,
                                               final NeuralNetConfiguration conf,
                                               final IActivation gateActivationFn,        //Activation function for the gates - sigmoid or hard sigmoid (must be found in range 0 to 1)
                                               final INDArray input,
                                               final INDArray recurrentWeights,      //Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
                                               final INDArray originalInputWeights,  //Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
                                               final INDArray biases,                //Shape: [4,hiddenLayerSize]; order: [bi,bf,bo,bg]^T
                                               final boolean training,
                                               final INDArray originalPrevOutputActivations,
                                               final INDArray originalPrevMemCellState,
                                               boolean forBackprop,
                                               boolean forwards,
                                               final String inputWeightKey,
                                               INDArray maskArray                   //Input mask: should only be used with bidirectional RNNs + variable length
                                               ) {

        //Mini-batch data format: for mini-batch size m, nIn inputs, and T time series length
        //Data has shape [m,nIn,T]. Layer activations/output has shape [m,nHiddenUnits,T]
        if(input == null || input.length() == 0) throw new IllegalArgumentException("Invalid input: not set or 0 length");

        INDArray inputWeights = originalInputWeights;
        INDArray prevOutputActivations = originalPrevOutputActivations;

        boolean is2dInput = input.rank() < 3;        //Edge case of T=1, may have shape [m,nIn], equiv. to [m,nIn,1]
        int timeSeriesLength = (is2dInput ? 1 : input.size(2));
        int hiddenLayerSize = recurrentWeights.size(0);
        int miniBatchSize = input.size(0);

        INDArray prevMemCellState;
        if (originalPrevMemCellState == null) {
            prevMemCellState = Nd4j.create(new int[]{miniBatchSize, hiddenLayerSize},'f');
        } else {
            prevMemCellState = originalPrevMemCellState.dup('f');
        }



        INDArray recurrentWeightsIFOG = recurrentWeights.get(NDArrayIndex.all(), NDArrayIndex.interval(0,4*hiddenLayerSize)).dup('f');


        //Apply dropconnect to input (not recurrent) weights only:
        if (conf.isUseDropConnect() && training && conf.getLayer().getDropOut() > 0) {
            inputWeights = Dropout.applyDropConnect(layer, inputWeightKey);
        }


        INDArray wFFTranspose = recurrentWeights.get(NDArrayIndex.all(), interval(4 * hiddenLayerSize, 4 * hiddenLayerSize + 1)).transpose(); //current
        INDArray wOOTranspose = recurrentWeights.get(NDArrayIndex.all(), interval(4 * hiddenLayerSize + 1, 4 * hiddenLayerSize + 2)).transpose(); //current
        INDArray wGGTranspose = recurrentWeights.get(NDArrayIndex.all(), interval(4 * hiddenLayerSize + 2, 4 * hiddenLayerSize + 3)).transpose(); //previous

        if (timeSeriesLength > 1 || forBackprop) {
            wFFTranspose = Shape.toMmulCompatible(wFFTranspose);
            wOOTranspose = Shape.toMmulCompatible(wOOTranspose);
            wGGTranspose = Shape.toMmulCompatible(wGGTranspose);
        }

        //Allocate arrays for activations:
        boolean sigmoidGates = gateActivationFn instanceof ActivationSigmoid;
        IActivation afn = conf.getLayer().getActivationFn();
        INDArray outputActivations = null;

        FwdPassReturn toReturn = new FwdPassReturn();
        if (forBackprop) {
            toReturn.fwdPassOutputAsArrays = new INDArray[timeSeriesLength];
            toReturn.memCellState = new INDArray[timeSeriesLength];
            toReturn.memCellActivations = new INDArray[timeSeriesLength];
            toReturn.iz = new INDArray[timeSeriesLength];
            toReturn.ia = new INDArray[timeSeriesLength];
            toReturn.fa = new INDArray[timeSeriesLength];
            toReturn.oa = new INDArray[timeSeriesLength];
            toReturn.ga = new INDArray[timeSeriesLength];
            if(!sigmoidGates){
                toReturn.fz = new INDArray[timeSeriesLength];
                toReturn.oz = new INDArray[timeSeriesLength];
                toReturn.gz = new INDArray[timeSeriesLength];
            }
        } else {
            outputActivations = Nd4j.create(new int[]{miniBatchSize, hiddenLayerSize, timeSeriesLength},'f');   //F order to keep time steps together
            toReturn.fwdPassOutput = outputActivations;
        }

        Level1 l1BLAS = Nd4j.getBlasWrapper().level1();

        //Input validation: check input data matches nIn
        if(input.size(1) != inputWeights.size(0)){
            throw new DL4JInvalidInputException("Received input with size(1) = " + input.size(1) + " (input array shape = "
                    + Arrays.toString(input.shape()) + "); input.size(1) must match layer nIn size (nIn = " + inputWeights.size(0) + ")");
        }
        //Input validation: check that if past state is provided, that it has same
        //These can be different if user forgets to call rnnClearPreviousState() between calls of rnnTimeStep
        if(prevOutputActivations != null && prevOutputActivations.size(0) != input.size(0)){
            throw new DL4JInvalidInputException("Previous activations (stored state) number of examples = " + prevOutputActivations.size(0)
                    + " but input array number of examples = " + input.size(0) + ". Possible cause: using rnnTimeStep() without calling"
                    + " rnnClearPreviousState() between different sequences?");
        }

        //initialize prevOutputActivations to zeroes
        if (prevOutputActivations == null) {
            prevOutputActivations = Nd4j.zeros(new int[]{miniBatchSize, hiddenLayerSize});
        }

        for (int iTimeIndex = 0; iTimeIndex < timeSeriesLength; iTimeIndex++) {
            int time = iTimeIndex;

            if (!forwards) {
                time = timeSeriesLength - iTimeIndex - 1;
            }


            INDArray miniBatchData = (is2dInput ? input : input.tensorAlongDimension(time, 1, 0));    //[Expected shape: [m,nIn]. Also deals with edge case of T=1, with 'time series' data of shape [m,nIn], equiv. to [m,nIn,1]
            miniBatchData = Shape.toMmulCompatible(miniBatchData);


            //Calculate activations for: network input + forget, output, input modulation gates. Next 3 lines are first part of those
            INDArray ifogActivations = miniBatchData.mmul(inputWeights);    //Shape: [miniBatch,4*layerSize]
            Nd4j.gemm(prevOutputActivations, recurrentWeightsIFOG, ifogActivations, false, false, 1.0, 1.0);
            ifogActivations.addiRowVector(biases);

            INDArray inputActivations = ifogActivations.get(NDArrayIndex.all(), NDArrayIndex.interval(0,hiddenLayerSize));
            if (forBackprop) toReturn.iz[time] = inputActivations.dup('f');
            conf.getLayer().getActivationFn().getActivation(inputActivations, training);
            if (forBackprop) toReturn.ia[time] = inputActivations;

            INDArray forgetGateActivations = ifogActivations.get(NDArrayIndex.all(), NDArrayIndex.interval(hiddenLayerSize,2*hiddenLayerSize));
            INDArray pmcellWFF = prevMemCellState.dup('f').muliRowVector(wFFTranspose);
            l1BLAS.axpy(pmcellWFF.length(), 1.0, pmcellWFF, forgetGateActivations);   //y = a*x + y i.e., forgetGateActivations.addi(pmcellWFF)
            //Above line: treats matrix as a vector. Can only do this because we're sure both pwcelWFF and forgetGateACtivations are f order, offset 0 and have same strides
            if(forBackprop && !sigmoidGates){
                toReturn.fz[time] = forgetGateActivations.dup('f');     //Forget gate pre-out (z)
            }
            gateActivationFn.getActivation(forgetGateActivations, training);

            if (forBackprop) toReturn.fa[time] = forgetGateActivations;


            INDArray inputModGateActivations = ifogActivations.get(NDArrayIndex.all(), NDArrayIndex.interval(3*hiddenLayerSize,4*hiddenLayerSize));
            INDArray pmcellWGG = prevMemCellState.dup('f').muliRowVector(wGGTranspose);
            l1BLAS.axpy(pmcellWGG.length(), 1.0, pmcellWGG, inputModGateActivations);   //inputModGateActivations.addi(pmcellWGG)
            if(forBackprop && !sigmoidGates){
                toReturn.gz[time] = inputModGateActivations.dup('f');   //Input modulation gate pre-out (z)
            }
            gateActivationFn.getActivation(inputModGateActivations, training);
            if (forBackprop) toReturn.ga[time] = inputModGateActivations;

            //Memory cell state
            INDArray currentMemoryCellState;
            INDArray inputModMulInput;
            if(forBackprop){
                currentMemoryCellState = prevMemCellState.dup('f').muli(forgetGateActivations);
                inputModMulInput = inputModGateActivations.dup('f').muli(inputActivations);
            } else {
                currentMemoryCellState = forgetGateActivations.muli(prevMemCellState);
                inputModMulInput = inputModGateActivations.muli(inputActivations);
            }
            l1BLAS.axpy(currentMemoryCellState.length(), 1.0, inputModMulInput, currentMemoryCellState);   //currentMemoryCellState.addi(inputModMulInput)

            INDArray outputGateActivations = ifogActivations.get(NDArrayIndex.all(), NDArrayIndex.interval(2*hiddenLayerSize,3*hiddenLayerSize));
            INDArray pmcellWOO = currentMemoryCellState.dup('f').muliRowVector(wOOTranspose);
            l1BLAS.axpy(pmcellWOO.length(), 1.0, pmcellWOO, outputGateActivations);   //outputGateActivations.addi(pmcellWOO)
            if(forBackprop && !sigmoidGates){
                toReturn.oz[time] = outputGateActivations.dup('f');     //Output gate activations
            }
            gateActivationFn.getActivation(outputGateActivations, training);
            if (forBackprop) toReturn.oa[time] = outputGateActivations;

            //LSTM unit outputs:
            INDArray currMemoryCellActivation = afn.getActivation(currentMemoryCellState.dup('f'), training);
            INDArray currHiddenUnitActivations;
            if(forBackprop){
                currHiddenUnitActivations = currMemoryCellActivation.dup('f').muli(outputGateActivations);    //Expected shape: [m,hiddenLayerSize]
            } else {
                currHiddenUnitActivations = currMemoryCellActivation.muli(outputGateActivations);    //Expected shape: [m,hiddenLayerSize]
            }

            if(maskArray != null){
                //Mask array is present: bidirectional RNN -> need to zero out these activations to avoid
                // incorrectly using activations from masked time steps (i.e., want 0 initialization in both directions)
                //We *also* need to apply this to the memory cells, as they are carried forward
                //Mask array has shape [minibatch, timeSeriesLength] -> get column
                INDArray timeStepMaskColumn = maskArray.getColumn(time);
                currHiddenUnitActivations.muliColumnVector(timeStepMaskColumn);
                currentMemoryCellState.muliColumnVector(timeStepMaskColumn);
            }

            if (forBackprop) {
                toReturn.fwdPassOutputAsArrays[time] = currHiddenUnitActivations;
                toReturn.memCellState[time] = currentMemoryCellState;
                toReturn.memCellActivations[time] = currMemoryCellActivation;
            } else {
                outputActivations.tensorAlongDimension(time, 1, 0).assign(currHiddenUnitActivations);
            }

            prevOutputActivations = currHiddenUnitActivations;
            prevMemCellState = currentMemoryCellState;

            toReturn.lastAct = currHiddenUnitActivations;
            toReturn.lastMemCell = currentMemoryCellState;
        }

        return toReturn;
    }

    static public Pair<Gradient, INDArray> backpropGradientHelper(final NeuralNetConfiguration conf,
                                                                  final IActivation gateActivationFn,
                                                                  final INDArray input,
                                                                  final INDArray recurrentWeights,      //Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
                                                                  final INDArray inputWeights,  //Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
                                                                  final INDArray epsilon,
                                                                  final boolean truncatedBPTT,
                                                                  final int tbpttBackwardLength,
                                                                  final FwdPassReturn fwdPass,
                                                                  final boolean forwards,
                                                                  final String inputWeightKey,
                                                                  final String recurrentWeightKey,
                                                                  final String biasWeightKey,
                                                                  final Map<String,INDArray> gradientViews,
                                                                  INDArray maskArray                   //Input mask: should only be used with bidirectional RNNs + variable length
                                                                  ) {


        //Expect errors to have shape: [miniBatchSize,n^(L+1),timeSeriesLength]
        int hiddenLayerSize = recurrentWeights.size(0);    //i.e., n^L
        int prevLayerSize = inputWeights.size(0);    //n^(L-1)
        int miniBatchSize = epsilon.size(0);
        boolean is2dInput = epsilon.rank() < 3; //Edge case: T=1 may have shape [miniBatchSize,n^(L+1)], equiv. to [miniBatchSize,n^(L+1),1]
        int timeSeriesLength = (is2dInput ? 1 : epsilon.size(2));

        INDArray wFFTranspose = recurrentWeights.get(NDArrayIndex.all(), point(4 * hiddenLayerSize)).transpose();
        INDArray wOOTranspose = recurrentWeights.get(NDArrayIndex.all(), point(4 * hiddenLayerSize+1)).transpose();
        INDArray wGGTranspose = recurrentWeights.get(NDArrayIndex.all(), point(4 * hiddenLayerSize+2)).transpose();

        INDArray wIFOG = recurrentWeights.get(NDArrayIndex.all(), NDArrayIndex.interval(0,4*hiddenLayerSize));
        //F order here so that content for time steps are together
        INDArray epsilonNext = Nd4j.create(new int[]{miniBatchSize, prevLayerSize, timeSeriesLength},'f');    //i.e., what would be W^L*(delta^L)^T. Shape: [m,n^(L-1),T]

        INDArray nablaCellStateNext = null;

        INDArray deltaifogNext = Nd4j.create(new int[]{miniBatchSize,4*hiddenLayerSize},'f');
        INDArray deltaiNext = deltaifogNext.get(NDArrayIndex.all(), NDArrayIndex.interval(0,hiddenLayerSize));
        INDArray deltafNext = deltaifogNext.get(NDArrayIndex.all(), NDArrayIndex.interval(hiddenLayerSize,2*hiddenLayerSize));
        INDArray deltaoNext = deltaifogNext.get(NDArrayIndex.all(), NDArrayIndex.interval(2*hiddenLayerSize,3*hiddenLayerSize));
        INDArray deltagNext = deltaifogNext.get(NDArrayIndex.all(), NDArrayIndex.interval(3*hiddenLayerSize,4*hiddenLayerSize));

        Level1 l1BLAS = Nd4j.getBlasWrapper().level1();
        int endIdx = 0;

        if (truncatedBPTT) {
            endIdx = Math.max(0, timeSeriesLength - tbpttBackwardLength);
        }

        //Get gradients. Note that we have to manually zero these, as they might not be initialized (or still has data from last iteration)
        //Also note that they are in f order (as per param initializer) so can be used in gemm etc
        INDArray iwGradientsOut = gradientViews.get(inputWeightKey);
        INDArray rwGradientsOut = gradientViews.get(recurrentWeightKey);    //Order: {I,F,O,G,FF,OO,GG}
        INDArray bGradientsOut = gradientViews.get(biasWeightKey);
        iwGradientsOut.assign(0);
        rwGradientsOut.assign(0);
        bGradientsOut.assign(0);

        INDArray rwGradientsIFOG = rwGradientsOut.get(NDArrayIndex.all(), NDArrayIndex.interval(0,4*hiddenLayerSize));
        INDArray rwGradientsFF = rwGradientsOut.get(NDArrayIndex.all(), NDArrayIndex.point(4 * hiddenLayerSize));
        INDArray rwGradientsOO = rwGradientsOut.get(NDArrayIndex.all(), NDArrayIndex.point(4 * hiddenLayerSize + 1));
        INDArray rwGradientsGG = rwGradientsOut.get(NDArrayIndex.all(), NDArrayIndex.point(4 * hiddenLayerSize + 2));

        boolean sigmoidGates = gateActivationFn instanceof ActivationSigmoid;
        IActivation afn = conf.getLayer().getActivationFn();

        INDArray timeStepMaskColumn = null;
        for (int iTimeIndex = timeSeriesLength - 1; iTimeIndex >= endIdx; iTimeIndex--) {
            int time = iTimeIndex;
            int inext = 1;

            if (!forwards) {
                time = timeSeriesLength - iTimeIndex - 1;
                inext = -1;
            }


            //First: calclate the components of nablaCellState that relies on the next time step deltas, so we can overwrite the deltas
            INDArray nablaCellState;
            if(iTimeIndex != timeSeriesLength -1){
                nablaCellState = deltafNext.dup('f').muliRowVector(wFFTranspose);
                l1BLAS.axpy(nablaCellState.length(), 1.0, deltagNext.dup('f').muliRowVector(wGGTranspose), nablaCellState);
            } else {
                nablaCellState = Nd4j.create(new int[]{miniBatchSize,hiddenLayerSize},'f');
            }

            INDArray prevMemCellState = (iTimeIndex == 0 ? null : fwdPass.memCellState[time - inext]);
            INDArray prevHiddenUnitActivation = (iTimeIndex == 0 ? null : fwdPass.fwdPassOutputAsArrays[time - inext]);
            INDArray currMemCellState = fwdPass.memCellState[time];


            //LSTM unit output errors (dL/d(a_out)); not to be confused with \delta=dL/d(z_out)
            INDArray epsilonSlice = (is2dInput ? epsilon : epsilon.tensorAlongDimension(time, 1, 0));        //(w^{L+1}*(delta^{(L+1)t})^T)^T or equiv.

            INDArray nablaOut = Shape.toOffsetZeroCopy(epsilonSlice, 'f'); //Shape: [m,n^L]
            if (iTimeIndex != timeSeriesLength - 1) {
                //if t == timeSeriesLength-1 then deltaiNext etc are zeros
                Nd4j.gemm(deltaifogNext, wIFOG, nablaOut, false, true, 1.0, 1.0);
            }

            //Output gate deltas:
            INDArray sigmahOfS = fwdPass.memCellActivations[time];
            INDArray ao = fwdPass.oa[time];

            //Normally would use zo.dup() in above line, but won't be using zo again (for this time step). Ditto for zf, zg, zi
            INDArray deltao = deltaoNext;
            Nd4j.getExecutioner().exec(new MulOp(nablaOut,sigmahOfS,deltao));
            if(sigmoidGates){
                INDArray sigmaoPrimeOfZo = Nd4j.getExecutioner().execAndReturn(new TimesOneMinus(ao.dup('f')));    //Equivalent to sigmoid deriv on zo
                deltao.muli(sigmaoPrimeOfZo);
            } else {
                deltao.assign(gateActivationFn.backprop(fwdPass.oz[time], deltao).getFirst());      //Deltao needs to be modified in-place
                //TODO: optimize (no assign)
            }

            //Memory cell error:
            INDArray temp = afn.backprop(currMemCellState.dup('f'), ao.muli(nablaOut)).getFirst();      //TODO activation functions with params
            l1BLAS.axpy(nablaCellState.length(), 1.0, temp, nablaCellState);
            INDArray deltaMulRowWOO = deltao.dup('f').muliRowVector(wOOTranspose);
            l1BLAS.axpy(nablaCellState.length(), 1.0, deltaMulRowWOO, nablaCellState); //nablaCellState.addi(deltao.mulRowVector(wOOTranspose));
            if (iTimeIndex != timeSeriesLength - 1) {
                INDArray nextForgetGateAs = fwdPass.fa[time + inext];
                int length = nablaCellState.length();
                l1BLAS.axpy(length, 1.0, nextForgetGateAs.muli(nablaCellStateNext), nablaCellState);       //nablaCellState.addi(nextForgetGateAs.mul(nablaCellStateNext))
            }
            nablaCellStateNext = nablaCellState;    //Store for use in next iteration

            //Forget gate delta:
            INDArray af = fwdPass.fa[time];
            INDArray deltaf = null;
            if (iTimeIndex > 0) {
                deltaf = deltafNext;
                if( sigmoidGates ) {
                    Nd4j.getExecutioner().exec(new TimesOneMinus(af, deltaf));
                    deltaf.muli(nablaCellState);
                    deltaf.muli(prevMemCellState);
                } else {
                    INDArray temp2 = nablaCellState.mul(prevMemCellState);
                    deltaf.assign(gateActivationFn.backprop(fwdPass.fz[time].dup('f'), temp2).getFirst());  //deltaf needs to be modified in-place
                    //TODO activation functions with params
                }
            }
            //Shape: [m,n^L]

            //Input modulation gate delta:
            INDArray ag = fwdPass.ga[time];
            INDArray ai = fwdPass.ia[time];
            INDArray deltag = deltagNext;
            if( sigmoidGates ){
                Nd4j.getExecutioner().exec(new TimesOneMinus(ag,deltag));   //Equivalent to sigmoid deriv on zg
                deltag.muli(ai);
                deltag.muli(nablaCellState);
            } else {
                INDArray temp2 = Nd4j.getExecutioner().execAndReturn(new MulOp(ai,nablaCellState,Nd4j.createUninitialized(ai.shape(), 'f')));
                deltag.assign(gateActivationFn.backprop(fwdPass.gz[time],temp2).getFirst());
                //TODO activation functions with params; optimize (no assign)
            }
            //Shape: [m,n^L]

            //Network input delta:
            INDArray zi = fwdPass.iz[time];
            INDArray deltai = deltaiNext;
            temp = Nd4j.getExecutioner().execAndReturn(new MulOp(ag,nablaCellState,Nd4j.createUninitialized(deltai.shape(), 'f')));
            deltai.assign(afn.backprop(zi, temp).getFirst());
            //TODO activation functions with params; also: optimize this (no assign)
            //Shape: [m,n^L]


            //Handle masking
            if (maskArray != null) {
                //Mask array is present: bidirectional RNN -> need to zero out these errors to avoid using errors from a masked time step
                // to calculate the parameter gradients.  Mask array has shape [minibatch, timeSeriesLength] -> get column(this time step)
                timeStepMaskColumn = maskArray.getColumn(time);
                deltaifogNext.muliColumnVector(timeStepMaskColumn);
                //Later, the deltaifogNext is used to calculate: input weight gradients, recurrent weight gradients, bias gradients
            }

            INDArray prevLayerActivationSlice = Shape.toMmulCompatible(is2dInput ? input : input.tensorAlongDimension(time, 1, 0));
            if(iTimeIndex > 0){
                //Again, deltaifog_current == deltaifogNext at this point... same array
                Nd4j.gemm(prevLayerActivationSlice, deltaifogNext, iwGradientsOut, true, false, 1.0, 1.0);
            } else {
                INDArray iwGradients_i = iwGradientsOut.get(NDArrayIndex.all(), NDArrayIndex.interval(0,hiddenLayerSize));
                Nd4j.gemm(prevLayerActivationSlice, deltai, iwGradients_i, true, false, 1.0, 1.0);
                INDArray iwGradients_og = iwGradientsOut.get(NDArrayIndex.all(), NDArrayIndex.interval(2*hiddenLayerSize, 4*hiddenLayerSize));
                INDArray deltaog = deltaifogNext.get(NDArrayIndex.all(), NDArrayIndex.interval(2*hiddenLayerSize,4*hiddenLayerSize));
                Nd4j.gemm(prevLayerActivationSlice, deltaog, iwGradients_og, true, false, 1.0, 1.0);
            }

            if (iTimeIndex > 0) {
                //If t==0, then prevHiddenUnitActivation==zeros(n^L,n^L), so dL/dW for recurrent weights will end up as 0 anyway
                //At this point: deltaifog and deltaifogNext are the same thing...
                //So what we are actually doing here is sum of (prevAct^transpose * deltaifog_current)
                Nd4j.gemm(prevHiddenUnitActivation, deltaifogNext, rwGradientsIFOG, true, false, 1.0, 1.0);

                //Shape: [1,n^L]. sum(0) is sum over examples in mini-batch.
                //Can use axpy here because result of sum and rwGradients[4 to 6] have order Nd4j.order(), via Nd4j.create()
                INDArray dLdwFF = deltaf.dup('f').muli(prevMemCellState).sum(0);    //mul not mmul because these weights are from unit j->j only (whereas other recurrent weights are i->j for all i,j)
                l1BLAS.axpy(hiddenLayerSize,1.0,dLdwFF,rwGradientsFF);     //rwGradients[4].addi(dLdwFF);    //dL/dw_{FF}
                INDArray dLdwGG = deltag.dup('f').muli(prevMemCellState).sum(0);
                l1BLAS.axpy(hiddenLayerSize,1.0,dLdwGG,rwGradientsGG);     //rwGradients[6].addi(dLdwGG);
            }

            INDArray dLdwOO = deltao.dup('f').muli(currMemCellState).sum(0);    //Expected shape: [n^L,1]. sum(0) is sum over examples in mini-batch.
            l1BLAS.axpy(hiddenLayerSize,1.0,dLdwOO,rwGradientsOO); //rwGradients[5].addi(dLdwOO);    //dL/dw_{OOxy}

            if(iTimeIndex > 0){
                l1BLAS.axpy(4*hiddenLayerSize,1.0, deltaifogNext.sum(0), bGradientsOut);
            } else {
                l1BLAS.axpy(hiddenLayerSize,1.0,deltai.sum(0),bGradientsOut);    //Sneaky way to do bGradients_i += deltai.sum(0)
                INDArray ogBiasToAdd = deltaifogNext.get(NDArrayIndex.all(), NDArrayIndex.interval(2*hiddenLayerSize,4*hiddenLayerSize)).sum(0);
                INDArray ogBiasGrad = bGradientsOut.get(NDArrayIndex.point(0), NDArrayIndex.interval(2*hiddenLayerSize,4*hiddenLayerSize));
                l1BLAS.axpy(2*hiddenLayerSize,1.0,ogBiasToAdd,ogBiasGrad);
            }

            //Calculate epsilonNext - i.e., equiv. to what would be (w^L*(d^(Lt))^T)^T in a normal network
            //But here, need to add 4 weights * deltas for the IFOG gates
            INDArray epsilonNextSlice = epsilonNext.tensorAlongDimension(time, 1, 0);   //This slice: f order and contiguous, due to epsilonNext being defined as f order.
            if(iTimeIndex > 0){
                Nd4j.gemm(deltaifogNext, inputWeights, epsilonNextSlice, false, true, 1.0, 1.0);
            } else {
                //No contribution from forget gate at t=0
                INDArray wi = inputWeights.get(NDArrayIndex.all(), NDArrayIndex.interval(0,hiddenLayerSize));
                Nd4j.gemm(deltai, wi, epsilonNextSlice, false, true, 1.0, 1.0);
                INDArray deltaog = deltaifogNext.get(NDArrayIndex.all(), NDArrayIndex.interval(2*hiddenLayerSize,4*hiddenLayerSize));
                INDArray wog = inputWeights.get(NDArrayIndex.all(), NDArrayIndex.interval(2*hiddenLayerSize,4*hiddenLayerSize));
                Nd4j.gemm(deltaog, wog, epsilonNextSlice, false, true, 1.0, 1.0);   //epsilonNextSlice.addi(deltao.mmul(woTranspose)).addi(deltag.mmul(wgTranspose));
            }

            if (maskArray != null) {
                //Mask array is present: bidirectional RNN -> need to zero out these errors to avoid sending anything
                // but 0s to the layer below at this time step (for the given example)
                epsilonNextSlice.muliColumnVector(timeStepMaskColumn);
            }
        }

        Gradient retGradient = new DefaultGradient();
        retGradient.gradientForVariable().put(inputWeightKey, iwGradientsOut);
        retGradient.gradientForVariable().put(recurrentWeightKey, rwGradientsOut);
        retGradient.gradientForVariable().put(biasWeightKey, bGradientsOut);

        return new Pair<>(retGradient, epsilonNext);
    }
}