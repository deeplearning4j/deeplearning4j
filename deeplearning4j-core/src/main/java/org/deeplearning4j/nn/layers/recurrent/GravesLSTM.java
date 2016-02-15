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
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * LSTM layer implementation.
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
    public Gradient calcGradient(Gradient layerError, INDArray activation) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        return backpropGradientHelper(epsilon, false, -1);
    }

    @Override
    public Pair<Gradient, INDArray> tbpttBackpropGradient(INDArray epsilon, int tbpttBackwardLength) {
        return backpropGradientHelper(epsilon, true, tbpttBackwardLength);
    }


    private Pair<Gradient, INDArray> backpropGradientHelper(final INDArray epsilon,final boolean truncatedBPTT,final int tbpttBackwardLength) {

        final INDArray inputWeights = getParam(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);
        final INDArray recurrentWeights = getParam(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY);    //Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]

        //First: Do forward pass to get gate activations, zs etc.
        FwdPassReturn fwdPass;
        if (truncatedBPTT) {
            fwdPass = activateHelper(true, stateMap.get(STATE_KEY_PREV_ACTIVATION), stateMap.get(STATE_KEY_PREV_MEMCELL), true);
            //Store last time step of output activations and memory cell state in tBpttStateMap
            tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct);
            tBpttStateMap.put(STATE_KEY_PREV_MEMCELL, fwdPass.lastMemCell);
        } else {
            fwdPass = activateHelper(true, null, null, true);
        }


        return LSTMHelpers.backpropGradientHelper(this.conf,
                this.input,
                recurrentWeights,
                inputWeights,
                epsilon,
                truncatedBPTT,
                tbpttBackwardLength,
                fwdPass,
                true,
                GravesLSTMParamInitializer.INPUT_WEIGHT_KEY,
                GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY,
                GravesLSTMParamInitializer.BIAS_KEY);
    }



    @Override
    public INDArray preOutput(INDArray x) {
        return activate(x, true);
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        return activate(x, training);
    }

    @Override
    public INDArray activate(INDArray input, boolean training) {
        setInput(input);
        return activateHelper(training, null, null, false).fwdPassOutput;
    }

    @Override
    public INDArray activate(INDArray input) {
        setInput(input);
        return activateHelper(true, null, null, false).fwdPassOutput;
    }

    @Override
    public INDArray activate(boolean training) {
        return activateHelper(training, null, null, false).fwdPassOutput;
    }

    @Override
    public INDArray activate() {

        return activateHelper(false, null, null, false).fwdPassOutput;
    }

    private FwdPassReturn activateHelper(final boolean training,
                                         final INDArray prevOutputActivations,
                                         final INDArray prevMemCellState,
                                         boolean forBackprop) {

        final INDArray recurrentWeights = getParam(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY);    //Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
        final INDArray inputWeights = getParam(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);            //Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
        final INDArray biases = getParam(GravesLSTMParamInitializer.BIAS_KEY); //by row: IFOG			//Shape: [4,hiddenLayerSize]; order: [bi,bf,bo,bg]^T

        return LSTMHelpers.activateHelper(this,this.conf,this.input,recurrentWeights,inputWeights,biases,training,prevOutputActivations,prevMemCellState,forBackprop,true,GravesLSTMParamInitializer.INPUT_WEIGHT_KEY);
    }

    @Override
    public INDArray activationMean() {
        return activate();
    }

    @Override
    public Type type() {
        return Type.RECURRENT;
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public double calcL2() {
        if (!conf.isUseRegularization() || conf.getLayer().getL2() <= 0.0) return 0.0;

        double l2Norm = getParam(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY).norm2Number().doubleValue();
        double sumSquaredWeights = l2Norm*l2Norm;

        l2Norm = getParam(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY).norm2Number().doubleValue();
        sumSquaredWeights += l2Norm*l2Norm;

        return 0.5 * conf.getLayer().getL2() * sumSquaredWeights;
    }

    @Override
    public double calcL1() {
        if (!conf.isUseRegularization() || conf.getLayer().getL1() <= 0.0) return 0.0;
        double l1 = getParam(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY).norm1Number().doubleValue()
                + getParam(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY).norm1Number().doubleValue();
        return conf.getLayer().getL1() * l1;
    }

    @Override
    public INDArray rnnTimeStep(INDArray input) {
        setInput(input);
        FwdPassReturn fwdPass = activateHelper(false, stateMap.get(STATE_KEY_PREV_ACTIVATION), stateMap.get(STATE_KEY_PREV_MEMCELL), false);
        INDArray outAct = fwdPass.fwdPassOutput;
        //Store last time step of output activations and memory cell state for later use:
        stateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct);
        stateMap.put(STATE_KEY_PREV_MEMCELL, fwdPass.lastMemCell);

        return outAct;
    }



    @Override
    public INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT) {
        setInput(input);
        FwdPassReturn fwdPass = activateHelper(training, stateMap.get(STATE_KEY_PREV_ACTIVATION), stateMap.get(STATE_KEY_PREV_MEMCELL), false);
        INDArray outAct = fwdPass.fwdPassOutput;
        if (storeLastForTBPTT) {
            //Store last time step of output activations and memory cell state in tBpttStateMap
            tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct);
            tBpttStateMap.put(STATE_KEY_PREV_MEMCELL, fwdPass.lastMemCell);
        }

        return outAct;
    }
}
