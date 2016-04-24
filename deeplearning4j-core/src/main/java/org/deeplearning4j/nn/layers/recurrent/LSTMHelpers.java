package org.deeplearning4j.nn.layers.recurrent;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

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
 * @author Alex Black
 * @author Benjamin Joseph
 */
public class LSTMHelpers {

    /**
     * Returns FwdPassReturn object with activations/INDArrays. Allows activateHelper to be used for forward pass, backward pass
     * and rnnTimeStep whilst being reasonably efficient for all
     */
    static public FwdPassReturn activateHelper( final Layer layer,
                                                final NeuralNetConfiguration conf,
                                                final INDArray input,
                                                final INDArray recurrentWeights,      //Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
                                                final INDArray originalInputWeights,  //Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
                                                final INDArray biases,                //Shape: [4,hiddenLayerSize]; order: [bi,bf,bo,bg]^T
                                                final boolean training,
                                                final INDArray originalPrevOutputActivations,
                                                final INDArray originalPrevMemCellState,
                                                boolean forBackprop,
                                                boolean forwards,
                                                final String inputWeightKey) {
        //Mini-batch data format: for mini-batch size m, nIn inputs, and T time series length
        //Data has shape [m,nIn,T]. Layer activations/output has shape [m,nHiddenUnits,T]
        if(input == null || input.length() == 0) throw new IllegalArgumentException("Invalid input: not set or 0 length");

        INDArray inputWeights = originalInputWeights;
        INDArray prevOutputActivations = originalPrevOutputActivations;
        INDArray prevMemCellState = originalPrevMemCellState;

        boolean is2dInput = input.rank() < 3;        //Edge case of T=1, may have shape [m,nIn], equiv. to [m,nIn,1]
        int timeSeriesLength = (is2dInput ? 1 : input.size(2));
        int hiddenLayerSize = recurrentWeights.size(0);
        int miniBatchSize = input.size(0);



        //Apply dropconnect to input (not recurrent) weights only:
        if (conf.isUseDropConnect() && training) {
            if (conf.getLayer().getDropOut() > 0) {
                inputWeights = Dropout.applyDropConnect(layer, inputWeightKey);
            }
        }

        //Extract weights and biases:
        INDArray wi = inputWeights.get(NDArrayIndex.all(), interval(0, hiddenLayerSize));    //i.e., want rows 0..nIn, columns 0..hiddenLayerSize
        INDArray wI = recurrentWeights.get(NDArrayIndex.all(), interval(0, hiddenLayerSize));
        INDArray bi = biases.get(NDArrayIndex.point(0), interval(0, hiddenLayerSize));

        INDArray wf = inputWeights.get(NDArrayIndex.all(), interval(hiddenLayerSize, 2 * hiddenLayerSize));
        INDArray wF = recurrentWeights.get(NDArrayIndex.all(), interval(hiddenLayerSize, 2 * hiddenLayerSize)); //previous
        INDArray wFFTranspose = recurrentWeights.get(NDArrayIndex.all(), interval(4 * hiddenLayerSize, 4 * hiddenLayerSize + 1)).transpose(); //current
        INDArray bf = biases.get(NDArrayIndex.point(0), interval(hiddenLayerSize, 2 * hiddenLayerSize));

        INDArray wo = inputWeights.get(NDArrayIndex.all(), interval(2 * hiddenLayerSize, 3 * hiddenLayerSize));
        INDArray wO = recurrentWeights.get(NDArrayIndex.all(), interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)); //previous
        INDArray wOOTranspose = recurrentWeights.get(NDArrayIndex.all(), interval(4 * hiddenLayerSize + 1, 4 * hiddenLayerSize + 2)).transpose(); //current
        INDArray bo = biases.get(NDArrayIndex.point(0), interval(2 * hiddenLayerSize, 3 * hiddenLayerSize));

        INDArray wg = inputWeights.get(NDArrayIndex.all(), interval(3 * hiddenLayerSize, 4 * hiddenLayerSize));
        INDArray wG = recurrentWeights.get(NDArrayIndex.all(), interval(3 * hiddenLayerSize, 4 * hiddenLayerSize)); //previous
        INDArray wGGTranspose = recurrentWeights.get(NDArrayIndex.all(), interval(4 * hiddenLayerSize + 2, 4 * hiddenLayerSize + 3)).transpose(); //previous
        INDArray bg = biases.get(NDArrayIndex.point(0), interval(3 * hiddenLayerSize, 4 * hiddenLayerSize));

        if (timeSeriesLength > 1 || forBackprop) {
            wi = Shape.toMmulCompatible(wi);
            wI = Shape.toMmulCompatible(wI);
            wf = Shape.toMmulCompatible(wf);
            wF = Shape.toMmulCompatible(wF);
            wFFTranspose = Shape.toMmulCompatible(wFFTranspose);
            wo = Shape.toMmulCompatible(wo);
            wO = Shape.toMmulCompatible(wO);
            wOOTranspose = Shape.toMmulCompatible(wOOTranspose);
            wg = Shape.toMmulCompatible(wg);
            wG = Shape.toMmulCompatible(wG);
            wGGTranspose = Shape.toMmulCompatible(wGGTranspose);
            bi = Shape.toMmulCompatible(bi);
            bf = Shape.toMmulCompatible(bf);
            bo = Shape.toMmulCompatible(bo);
            bg = Shape.toMmulCompatible(bg);
        }

        //Allocate arrays for activations:
        INDArray outputActivations = null;

        FwdPassReturn toReturn = new FwdPassReturn();
        if (forBackprop) {
            toReturn.paramsMmulCompatible = new INDArray[]{wi, wI, wf, wF, wFFTranspose, wo, wO, wOOTranspose, wg, wG, wGGTranspose};
            toReturn.fwdPassOutputAsArrays = new INDArray[timeSeriesLength];
            toReturn.memCellState = new INDArray[timeSeriesLength];
            toReturn.memCellActivations = new INDArray[timeSeriesLength];
            toReturn.iz = new INDArray[timeSeriesLength];
            toReturn.ia = new INDArray[timeSeriesLength];
            toReturn.fa = new INDArray[timeSeriesLength];
            toReturn.oa = new INDArray[timeSeriesLength];
            toReturn.ga = new INDArray[timeSeriesLength];
        } else {
//            outputActivations = Nd4j.zeros(new int[]{miniBatchSize, hiddenLayerSize, timeSeriesLength});      //Before
            outputActivations = Nd4j.create(new int[]{miniBatchSize, hiddenLayerSize, timeSeriesLength},'f');   //F order to keep time steps together
            toReturn.fwdPassOutput = outputActivations;
        }

        Level1 l1BLAS = Nd4j.getBlasWrapper().level1();

        //initialize prevOutputActivations to zeroes
        if (prevOutputActivations == null) {
            prevOutputActivations = Nd4j.zeros(new int[]{miniBatchSize, hiddenLayerSize});
        }

        if (prevMemCellState == null) {
            prevMemCellState = Nd4j.zeros(new int[]{miniBatchSize, hiddenLayerSize});
        }

        for (int iTimeIndex = 0; iTimeIndex < timeSeriesLength; iTimeIndex++) {
            int time = iTimeIndex;

            if (!forwards) {
                time = timeSeriesLength - iTimeIndex - 1;
            }


            INDArray miniBatchData = (is2dInput ? input : input.tensorAlongDimension(time, 1, 0));    //[Expected shape: [m,nIn]. Also deals with edge case of T=1, with 'time series' data of shape [m,nIn], equiv. to [m,nIn,1]
            miniBatchData = Shape.toMmulCompatible(miniBatchData);

//            //Calculate activations for: network input + forget, output, input modulation gates.
//            INDArray inputActivations = miniBatchData.mmul(wi);
//            Nd4j.gemm(prevOutputActivations, wI, inputActivations, false, false, 1.0, 1.0);
//            inputActivations.addiRowVector(bi);
//            if (forBackprop) toReturn.iz[time] = inputActivations.dup('f');
//            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), inputActivations));
//            if (forBackprop) toReturn.ia[time] = inputActivations;
//
//            INDArray forgetGateActivations = miniBatchData.mmul(wf);
//            Nd4j.gemm(prevOutputActivations, wF, forgetGateActivations, false, false, 1.0, 1.0);
//            INDArray pmcellWFF = prevMemCellState.dup('f').muliRowVector(wFFTranspose);
//            l1BLAS.axpy(pmcellWFF.length(), 1.0, pmcellWFF, forgetGateActivations);   //y = a*x + y i.e., forgetGateActivations.addi(pmcellWFF)
//            //Above line: treats matrix as a vector. Can only do this because we're sure both pwcelWFF and forgetGateACtivations are f order, offset 0 and have same strides
//            forgetGateActivations.addiRowVector(bf);
//            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", forgetGateActivations));
//            if (forBackprop) toReturn.fa[time] = forgetGateActivations;
//
//
//            INDArray inputModGateActivations = miniBatchData.mmul(wg);
//            Nd4j.gemm(prevOutputActivations, wG, inputModGateActivations, false, false, 1.0, 1.0);
//            INDArray pmcellWGG = prevMemCellState.dup('f').muliRowVector(wGGTranspose);
//            l1BLAS.axpy(pmcellWGG.length(), 1.0, pmcellWGG, inputModGateActivations);   //inputModGateActivations.addi(pmcellWGG)
//            inputModGateActivations.addiRowVector(bg);
//            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", inputModGateActivations));
//            if (forBackprop) toReturn.ga[time] = inputModGateActivations;
//
//            //Memory cell state
//            INDArray currentMemoryCellState = forgetGateActivations.dup('f').muli(prevMemCellState);
//            INDArray inputModMulInput = inputModGateActivations.dup('f').muli(inputActivations);
//            l1BLAS.axpy(currentMemoryCellState.length(), 1.0, inputModMulInput, currentMemoryCellState);   //currentMemoryCellState.addi(inputModMulInput)
//
//            INDArray outputGateActivations = miniBatchData.mmul(wo);
//            Nd4j.gemm(prevOutputActivations, wO, outputGateActivations, false, false, 1.0, 1.0);
//            INDArray pmcellWOO = currentMemoryCellState.dup('f').muliRowVector(wOOTranspose);
//            l1BLAS.axpy(pmcellWOO.length(), 1.0, pmcellWOO, outputGateActivations);   //outputGateActivations.addi(pmcellWOO)
//            outputGateActivations.addiRowVector(bo);
//            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", outputGateActivations));
//            if (forBackprop) toReturn.oa[time] = outputGateActivations;


            //Calculate activations for: network input + forget, output, input modulation gates.
            INDArray ifogActivations = miniBatchData.mmul(inputWeights);    //Shape: [miniBatch,4*layerSize]

//            INDArray inputActivations = miniBatchData.mmul(wi);
            INDArray inputActivations = ifogActivations.get(NDArrayIndex.all(), NDArrayIndex.interval(0,hiddenLayerSize));
            Nd4j.gemm(prevOutputActivations, wI, inputActivations, false, false, 1.0, 1.0);
            inputActivations.addiRowVector(bi);
            if (forBackprop) toReturn.iz[time] = inputActivations.dup('f');
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), inputActivations));
            if (forBackprop) toReturn.ia[time] = inputActivations;

//            INDArray forgetGateActivations = miniBatchData.mmul(wf);
            INDArray forgetGateActivations = ifogActivations.get(NDArrayIndex.all(), NDArrayIndex.interval(hiddenLayerSize,2*hiddenLayerSize));
            Nd4j.gemm(prevOutputActivations, wF, forgetGateActivations, false, false, 1.0, 1.0);
            INDArray pmcellWFF = prevMemCellState.dup('f').muliRowVector(wFFTranspose);
            l1BLAS.axpy(pmcellWFF.length(), 1.0, pmcellWFF, forgetGateActivations);   //y = a*x + y i.e., forgetGateActivations.addi(pmcellWFF)
            //Above line: treats matrix as a vector. Can only do this because we're sure both pwcelWFF and forgetGateACtivations are f order, offset 0 and have same strides
            forgetGateActivations.addiRowVector(bf);
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", forgetGateActivations));
            if (forBackprop) toReturn.fa[time] = forgetGateActivations;


//            INDArray inputModGateActivations = miniBatchData.mmul(wg);
            INDArray inputModGateActivations = ifogActivations.get(NDArrayIndex.all(), NDArrayIndex.interval(3*hiddenLayerSize,4*hiddenLayerSize));
            Nd4j.gemm(prevOutputActivations, wG, inputModGateActivations, false, false, 1.0, 1.0);
            INDArray pmcellWGG = prevMemCellState.dup('f').muliRowVector(wGGTranspose);
            l1BLAS.axpy(pmcellWGG.length(), 1.0, pmcellWGG, inputModGateActivations);   //inputModGateActivations.addi(pmcellWGG)
            inputModGateActivations.addiRowVector(bg);
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", inputModGateActivations));
            if (forBackprop) toReturn.ga[time] = inputModGateActivations;

            //Memory cell state
            INDArray currentMemoryCellState = forgetGateActivations.dup('f').muli(prevMemCellState);
            INDArray inputModMulInput = inputModGateActivations.dup('f').muli(inputActivations);
            l1BLAS.axpy(currentMemoryCellState.length(), 1.0, inputModMulInput, currentMemoryCellState);   //currentMemoryCellState.addi(inputModMulInput)

//            INDArray outputGateActivations = miniBatchData.mmul(wo);
            INDArray outputGateActivations = ifogActivations.get(NDArrayIndex.all(), NDArrayIndex.interval(2*hiddenLayerSize,3*hiddenLayerSize));
            Nd4j.gemm(prevOutputActivations, wO, outputGateActivations, false, false, 1.0, 1.0);
            INDArray pmcellWOO = currentMemoryCellState.dup('f').muliRowVector(wOOTranspose);
            l1BLAS.axpy(pmcellWOO.length(), 1.0, pmcellWOO, outputGateActivations);   //outputGateActivations.addi(pmcellWOO)
            outputGateActivations.addiRowVector(bo);
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("sigmoid", outputGateActivations));
            if (forBackprop) toReturn.oa[time] = outputGateActivations;

            //LSTM unit outputs:
            INDArray currMemoryCellActivation = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), currentMemoryCellState.dup('f')));
            INDArray currHiddenUnitActivations = currMemoryCellActivation.dup('f').muli(outputGateActivations);    //Expected shape: [m,hiddenLayerSize]

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
                                                                  final String biasWeightKey) {


        //Expect errors to have shape: [miniBatchSize,n^(L+1),timeSeriesLength]
        int hiddenLayerSize = recurrentWeights.size(0);    //i.e., n^L
        int prevLayerSize = inputWeights.size(0);    //n^(L-1)
        int miniBatchSize = epsilon.size(0);
        boolean is2dInput = epsilon.rank() < 3; //Edge case: T=1 may have shape [miniBatchSize,n^(L+1)], equiv. to [miniBatchSize,n^(L+1),1]
        int timeSeriesLength = (is2dInput ? 1 : epsilon.size(2));

        INDArray wi = fwdPass.paramsMmulCompatible[0];
        INDArray wI = fwdPass.paramsMmulCompatible[1];
        INDArray wf = fwdPass.paramsMmulCompatible[2];
        INDArray wF = fwdPass.paramsMmulCompatible[3];
        INDArray wo = fwdPass.paramsMmulCompatible[5];
        INDArray wO = fwdPass.paramsMmulCompatible[6];
        INDArray wg = fwdPass.paramsMmulCompatible[8];
        INDArray wG = fwdPass.paramsMmulCompatible[9];
        INDArray wFFTranspose = fwdPass.paramsMmulCompatible[4];
        INDArray wOOTranspose = fwdPass.paramsMmulCompatible[7];
        INDArray wGGTranspose = fwdPass.paramsMmulCompatible[10];

        //Parameter gradients, summed across time. bias gradients, input weight gradients, recurrent weight gradients
        INDArray[] bGradients = new INDArray[4];
        INDArray[] iwGradients = new INDArray[4];
        INDArray[] rwGradients = new INDArray[7];    //Order: {I,F,O,G,FF,OO,GG}
        for (int i = 0; i < 4; i++) {
            bGradients[i] = Nd4j.create(new int[]{1, hiddenLayerSize}); //Order as per Nd4j.order()
            iwGradients[i] = Nd4j.create(new int[]{prevLayerSize, hiddenLayerSize}, 'f'); //f order for use in gemm
            rwGradients[i] = Nd4j.create(new int[]{hiddenLayerSize, hiddenLayerSize}, 'f');
        }
        for (int i = 0; i < 3; i++) rwGradients[i + 4] = Nd4j.zeros(1, hiddenLayerSize);    //Order as per Nd4j.order()

        //Original:
        //INDArray epsilonNext = Nd4j.zeros(miniBatchSize, prevLayerSize, timeSeriesLength);    //i.e., what would be W^L*(delta^L)^T. Shape: [m,n^(L-1),T]

        //F order here so that content for time steps are together
        INDArray epsilonNext = Nd4j.create(new int[]{miniBatchSize, prevLayerSize, timeSeriesLength},'f');    //i.e., what would be W^L*(delta^L)^T. Shape: [m,n^(L-1),T]

        INDArray nablaCellStateNext = null;
        INDArray deltaiNext = null;
        INDArray deltafNext = null;
        INDArray deltaoNext = null;
        INDArray deltagNext = null;

        Level1 l1BLAS = Nd4j.getBlasWrapper().level1();
        int endIdx = 0;

        if (truncatedBPTT) {
            endIdx = Math.max(0, timeSeriesLength - tbpttBackwardLength);
        }

        for (int iTimeIndex = timeSeriesLength - 1; iTimeIndex >= endIdx; iTimeIndex--) {
            int time = iTimeIndex;
            int inext = 1;

            if (!forwards) {
                time = timeSeriesLength - iTimeIndex - 1;
                inext = -1;
            }

            INDArray prevMemCellState = (iTimeIndex == 0 ? null : fwdPass.memCellState[time - inext]);
            INDArray prevHiddenUnitActivation = (iTimeIndex == 0 ? null : fwdPass.fwdPassOutputAsArrays[time - inext]);
            INDArray currMemCellState = fwdPass.memCellState[time];


            //LSTM unit output errors (dL/d(a_out)); not to be confused with \delta=dL/d(z_out)
            INDArray epsilonSlice = (is2dInput ? epsilon : epsilon.tensorAlongDimension(time, 1, 0));        //(w^{L+1}*(delta^{(L+1)t})^T)^T or equiv.
            INDArray nablaOut = Shape.toOffsetZeroCopy(epsilonSlice, 'f'); //Shape: [m,n^L]
            if (iTimeIndex != timeSeriesLength - 1) {
                //if t == timeSeriesLength-1 then deltaiNext etc are zeros
                Nd4j.gemm(deltaiNext, wI, nablaOut, false, true, 1.0, 1.0);   //nablaOut.addi(deltaiNext.mmul(wITranspose))
                Nd4j.gemm(deltafNext, wF, nablaOut, false, true, 1.0, 1.0);   //nablaOut.addi(deltafNext.mmul(wFTranspose))
                Nd4j.gemm(deltaoNext, wO, nablaOut, false, true, 1.0, 1.0);   //nablaOut.addi(deltaoNext.mmul(wOTranspose))
                Nd4j.gemm(deltagNext, wG, nablaOut, false, true, 1.0, 1.0);   //nablaOut.addi(deltagNext.mmul(wGTranspose));
            }

            //Output gate deltas:
            INDArray sigmahOfS = fwdPass.memCellActivations[time];
            INDArray ao = fwdPass.oa[time];
            INDArray sigmaoPrimeOfZo = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("timesoneminus", ao.dup('f')));    //Equivalent to sigmoid deriv on zo
            //Normally would use zo.dup() in above line, but won't be using zo again (for this time step). Ditto for zf, zg, zi
            INDArray deltao = nablaOut.dup('f').muli(sigmahOfS).muli(sigmaoPrimeOfZo); //Shape: [m,n^L]

            //Memory cell error:
            INDArray sigmahPrimeOfS = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), currMemCellState.dup('f')).derivative());//	shape: [m,n^L]
            INDArray nablaCellState = ao.muli(nablaOut).muli(sigmahPrimeOfS);
            INDArray deltaMulRowWOO = deltao.dup('f').muliRowVector(wOOTranspose);
            l1BLAS.axpy(nablaCellState.length(), 1.0, deltaMulRowWOO, nablaCellState); //nablaCellState.addi(deltao.mulRowVector(wOOTranspose));
            if (iTimeIndex != timeSeriesLength - 1) {
                INDArray nextForgetGateAs = fwdPass.fa[time + inext];
                int length = nablaCellState.length();
                l1BLAS.axpy(length, 1.0, nextForgetGateAs.muli(nablaCellStateNext), nablaCellState);       //nablaCellState.addi(nextForgetGateAs.mul(nablaCellStateNext))
                l1BLAS.axpy(length, 1.0, deltafNext.dup('f').muliRowVector(wFFTranspose), nablaCellState);    //nablaCellState.addi(deltafNext.mulRowVector(wFFTranspose))
                l1BLAS.axpy(length, 1.0, deltagNext.dup('f').muliRowVector(wGGTranspose), nablaCellState);   //nablaCellState.addi(deltagNext.mulRowVector(wGGTranspose));
            }
            nablaCellStateNext = nablaCellState;    //Store for use in next iteration

            //Forget gate delta:
            INDArray af = fwdPass.fa[time];
            INDArray deltaf = null;
            if (iTimeIndex > 0) {
                deltaf = nablaCellState.dup('f').muli(prevMemCellState)
                        .muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("timesoneminus", af.dup('f'))));    //Equivalent to sigmoid deriv on zf
            }
            //Shape: [m,n^L]

            //Input modulation gate delta:
            INDArray ag = fwdPass.ga[time];
            INDArray ai = fwdPass.ia[time];
            INDArray deltag = ai.muli(nablaCellState)
                    .muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("timesoneminus", ag.dup('f'))));    //Equivalent to sigmoid deriv on zg
            //Shape: [m,n^L]

            //Network input delta:
            INDArray zi = fwdPass.iz[time];
            INDArray deltai = ag.muli(nablaCellState)
                    .muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), zi).derivative()));
            //Shape: [m,n^L]

            INDArray prevLayerActivationSlice = Shape.toMmulCompatible(is2dInput ? input : input.tensorAlongDimension(time, 1, 0));
            Nd4j.gemm(prevLayerActivationSlice, deltai, iwGradients[0], true, false, 1.0, 1.0);   //iwGradients[0].addi(prevLayerActivationSliceTransposed.mmul(deltai));

            if (iTimeIndex > 0) {
                Nd4j.gemm(prevLayerActivationSlice, deltaf, iwGradients[1], true, false, 1.0, 1.0);   //iwGradients[1].addi(prevLayerActivationSliceTransposed.mmul(deltaf));
            }

            Nd4j.gemm(prevLayerActivationSlice, deltao, iwGradients[2], true, false, 1.0, 1.0);   //iwGradients[2].addi(prevLayerActivationSliceTransposed.mmul(deltao));
            Nd4j.gemm(prevLayerActivationSlice, deltag, iwGradients[3], true, false, 1.0, 1.0);   //iwGradients[3].addi(prevLayerActivationSliceTransposed.mmul(deltag));

            if (iTimeIndex > 0) {
                //If t==0, then prevHiddenUnitActivation==zeros(n^L,n^L), so dL/dW for recurrent weights will end up as 0 anyway
                Nd4j.gemm(prevHiddenUnitActivation, deltai, rwGradients[0], true, false, 1.0, 1.0);   //rwGradients[0].addi(prevActTranspose.mmul(deltai));
                Nd4j.gemm(prevHiddenUnitActivation, deltaf, rwGradients[1], true, false, 1.0, 1.0);   //rwGradients[1].addi(prevActTranspose.mmul(deltaf));
                Nd4j.gemm(prevHiddenUnitActivation, deltao, rwGradients[2], true, false, 1.0, 1.0);   //rwGradients[2].addi(prevActTranspose.mmul(deltao));
                Nd4j.gemm(prevHiddenUnitActivation, deltag, rwGradients[3], true, false, 1.0, 1.0);   //rwGradients[3].addi(prevActTranspose.mmul(deltag));

                //Shape: [1,n^L]. sum(0) is sum over examples in mini-batch.
                //Can use axpy here because result of sum and rwGradients[4 to 6] have order Nd4j.order(), via Nd4j.create()
                INDArray dLdwFF = deltaf.dup('f').muli(prevMemCellState).sum(0);    //mul not mmul because these weights are from unit j->j only (whereas other recurrent weights are i->j for all i,j)
                l1BLAS.axpy(rwGradients[4].length(),1.0,dLdwFF,rwGradients[4]);     //rwGradients[4].addi(dLdwFF);    //dL/dw_{FF}
                INDArray dLdwGG = deltag.dup('f').muli(prevMemCellState).sum(0);
                l1BLAS.axpy(rwGradients[6].length(),1.0,dLdwGG,rwGradients[6]);     //rwGradients[6].addi(dLdwGG);
            }

            INDArray dLdwOO = deltao.dup('f').muli(currMemCellState).sum(0);    //Expected shape: [n^L,1]. sum(0) is sum over examples in mini-batch.
            l1BLAS.axpy(rwGradients[5].length(),1.0,dLdwOO,rwGradients[5]); //rwGradients[5].addi(dLdwOO);    //dL/dw_{OOxy}

            //Can use axpy here because result of sum and bGradients[i] both have order Nd4j.order(), via Nd4j.create()
            l1BLAS.axpy(bGradients[0].length(),1.0,deltai.sum(0),bGradients[0]);    //bGradients[0].addi(deltai.sum(0));
            if(iTimeIndex > 0) {
                l1BLAS.axpy(bGradients[1].length(),1.0,deltaf.sum(0),bGradients[1]);    //bGradients[1].addi(deltaf.sum(0));
            }

            l1BLAS.axpy(bGradients[2].length(),1.0,deltao.sum(0),bGradients[2]);    //bGradients[2].addi(deltao.sum(0));
            l1BLAS.axpy(bGradients[3].length(),1.0,deltag.sum(0),bGradients[3]);    //bGradients[3].addi(deltag.sum(0));

            //Calculate epsilonNext - i.e., equiv. to what would be (w^L*(d^(Lt))^T)^T in a normal network
            //But here, need to add 4 weights * deltas for the IFOG gates
            INDArray epsilonNextSlice = Nd4j.gemm(deltai,wi,false,true);
            Nd4j.gemm(deltao, wo, epsilonNextSlice, false, true, 1.0, 1.0);   //epsilonNextSlice.addi(deltao.mmul(woTranspose))
            Nd4j.gemm(deltag, wg, epsilonNextSlice, false, true, 1.0, 1.0);   //epsilonNextSlice.addi(deltag.mmul(wgTranspose));

            if (iTimeIndex > 0) {
                Nd4j.gemm(deltaf, wf, epsilonNextSlice, false, true, 1.0, 1.0); //epsilonNextSlice.addi(deltaf.mmul(wfTranspose));
            }

            epsilonNext.tensorAlongDimension(time, 1, 0).assign(epsilonNextSlice);

            deltaiNext = deltai;
            deltafNext = deltaf;
            deltaoNext = deltao;
            deltagNext = deltag;
        }

        //Weight/bias gradients
        INDArray iwGradientsOut = Nd4j.zeros(prevLayerSize, 4 * hiddenLayerSize);
        INDArray rwGradientsOut = Nd4j.zeros(hiddenLayerSize, 4 * hiddenLayerSize + 3);    //Order: {I,F,O,G,FF,OO,GG}
        INDArray bGradientsOut = Nd4j.hstack(bGradients);
        iwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(), interval(0, hiddenLayerSize)}, iwGradients[0]);
        iwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(), interval(hiddenLayerSize, 2 * hiddenLayerSize)}, iwGradients[1]);
        iwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(), interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)}, iwGradients[2]);
        iwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(), interval(3 * hiddenLayerSize, 4 * hiddenLayerSize)}, iwGradients[3]);

        rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(), interval(0, hiddenLayerSize)}, rwGradients[0]);
        rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(), interval(hiddenLayerSize, 2 * hiddenLayerSize)}, rwGradients[1]);
        rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(), interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)}, rwGradients[2]);
        rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(), interval(3 * hiddenLayerSize, 4 * hiddenLayerSize)}, rwGradients[3]);
        rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4 * hiddenLayerSize)}, rwGradients[4].transpose());
        rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4 * hiddenLayerSize + 1)}, rwGradients[5].transpose());
        rwGradientsOut.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4 * hiddenLayerSize + 2)}, rwGradients[6].transpose());

        Gradient retGradient = new DefaultGradient();
        retGradient.gradientForVariable().put(inputWeightKey, iwGradientsOut);
        retGradient.gradientForVariable().put(recurrentWeightKey, rwGradientsOut);
        retGradient.gradientForVariable().put(biasWeightKey, bGradientsOut);

        return new Pair<>(retGradient, epsilonNext);
    }

}
