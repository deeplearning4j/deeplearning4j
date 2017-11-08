/*-
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

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.GravesLSTMParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * LSTM layer implementation.
 * Based on Graves: Supervised Sequence Labelling with Recurrent Neural Networks
 * http://www.cs.toronto.edu/~graves/phd.pdf
 * See also for full/vectorized equations (and a comparison to other LSTM variants):
 * Greff et al. 2015, "LSTM: A Search Space Odyssey", pg11. This is the "vanilla" variant in said paper
 * http://arxiv.org/pdf/1503.04069.pdf
 *
 * @author Alex Black
 * @see LSTM LSTM class, for the version without peephole connections
 */
@Slf4j
public class GravesLSTM extends BaseRecurrentLayer<org.deeplearning4j.nn.conf.layers.GravesLSTM> {
    public static final String STATE_KEY_PREV_ACTIVATION = "prevAct";
    public static final String STATE_KEY_PREV_MEMCELL = "prevMem";

    protected FwdPassReturn cachedFwdPass;

    public GravesLSTM(org.deeplearning4j.nn.conf.layers.GravesLSTM conf) {
        super(conf);
    }
    @Override
    public Gradients backpropGradient(Gradients epsilon) {
        return backpropGradientHelper(epsilon, false, -1);
    }

    @Override
    public Gradients tbpttBackpropGradient(Gradients epsilon, int tbpttBackwardLength) {
        return backpropGradientHelper(epsilon, true, tbpttBackwardLength);
    }


    private Gradients backpropGradientHelper(final Gradients gradients, final boolean truncatedBPTT,
                    final int tbpttBackwardLength) {
        INDArray epsilon = gradients.get(0);

        final INDArray inputWeights = getParamWithNoise(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY, true);
        final INDArray recurrentWeights = getParamWithNoise(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY, true); //Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]

        //First: Do forward pass to get gate activations, zs etc.
        FwdPassReturn fwdPass;
        if (truncatedBPTT) {
            fwdPass = activateHelper(true, stateMap.get(STATE_KEY_PREV_ACTIVATION),
                            stateMap.get(STATE_KEY_PREV_MEMCELL), true);
            //Store last time step of output activations and memory cell state in tBpttStateMap
            tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct.leverageTo(ComputationGraph.workspaceTBPTT));
            tBpttStateMap.put(STATE_KEY_PREV_MEMCELL, fwdPass.lastMemCell.leverageTo(ComputationGraph.workspaceTBPTT));
        } else {
            fwdPass = activateHelper(true, null, null, true);
        }


        //Note: masks should always be null, even when present: see comments in activate method
        Gradients p = LSTMHelpers.backpropGradientHelper(this.conf, this.layerConf().getGateActivationFn(), this.input.get(0),
                        recurrentWeights, inputWeights, epsilon, truncatedBPTT, tbpttBackwardLength, fwdPass, true,
                        GravesLSTMParamInitializer.INPUT_WEIGHT_KEY, GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY,
                        GravesLSTMParamInitializer.BIAS_KEY, gradientViews, null, true, null);

        weightNoiseParams.clear();
        return backpropPreprocessor(p);
    }

    @Override
    public Activations activate(Activations input, boolean training) {
        setInput(input);
        return activate(training);
    }

    @Override
    public Activations activate(boolean training) {
        INDArray ret = activateHelper(training, null, null, false).fwdPassOutput;

        //LSTM (standard, not bi-directional) don't make any changes to the data OR the mask arrays
        //Any relevant masking occurs during backprop
        //They also set the current mask array as inactive: this is for situations like the following:
        // in -> dense -> lstm -> dense -> lstm
        // The first dense should be masked using the input array, but the second shouldn't. If necessary, the second
        // dense will be masked via the output layer mask
        return ActivationsFactory.getInstance().create(ret, input.getMask(0), MaskState.Passthrough);
    }

    private FwdPassReturn activateHelper(final boolean training, final INDArray prevOutputActivations,
                    final INDArray prevMemCellState, boolean forBackprop) {
        applyPreprocessorIfNecessary(training);

        if (cacheMode == null)
            cacheMode = CacheMode.NONE;

        if (forBackprop && cachedFwdPass != null) {
            FwdPassReturn ret = cachedFwdPass;
            cachedFwdPass = null;
            return ret;
        }

        final INDArray recurrentWeights = getParamWithNoise(GravesLSTMParamInitializer.RECURRENT_WEIGHT_KEY, training); //Shape: [hiddenLayerSize,4*hiddenLayerSize+3]; order: [wI,wF,wO,wG,wFF,wOO,wGG]
        final INDArray inputWeights = getParamWithNoise(GravesLSTMParamInitializer.INPUT_WEIGHT_KEY, training); //Shape: [n^(L-1),4*hiddenLayerSize]; order: [wi,wf,wo,wg]
        final INDArray biases = getParamWithNoise(GravesLSTMParamInitializer.BIAS_KEY, training); //by row: IFOG			//Shape: [4,hiddenLayerSize]; order: [bi,bf,bo,bg]^T

        //Note: masks should always be null, even when present: see comments in activate method
        FwdPassReturn fwd = LSTMHelpers.activateHelper(this, this.conf, this.layerConf().getGateActivationFn(),
                        this.input.get(0), recurrentWeights, inputWeights, biases, training, prevOutputActivations,
                        prevMemCellState, forBackprop || (cacheMode != CacheMode.NONE && training), true,
                        GravesLSTMParamInitializer.INPUT_WEIGHT_KEY, null, true, null,
                        forBackprop ? cacheMode : CacheMode.NONE);


        if (training && cacheMode != CacheMode.NONE) {
            cachedFwdPass = fwd;
        }

        return fwd;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public Activations rnnTimeStep(Activations input) {
        setInput(input);
        FwdPassReturn fwdPass = activateHelper(false, stateMap.get(STATE_KEY_PREV_ACTIVATION),
                        stateMap.get(STATE_KEY_PREV_MEMCELL), false);
        INDArray outAct = fwdPass.fwdPassOutput;
        //Store last time step of output activations and memory cell state for later use:
        stateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct);
        stateMap.put(STATE_KEY_PREV_MEMCELL, fwdPass.lastMemCell);

        return ActivationsFactory.getInstance().create(outAct);
    }



    @Override
    public Activations rnnActivateUsingStoredState(Activations input, boolean training, boolean storeLastForTBPTT) {
        setInput(input);
        FwdPassReturn fwdPass = activateHelper(training, stateMap.get(STATE_KEY_PREV_ACTIVATION),
                        stateMap.get(STATE_KEY_PREV_MEMCELL), false);
        INDArray outAct = fwdPass.fwdPassOutput;
        if (storeLastForTBPTT) {
            //Store last time step of output activations and memory cell state in tBpttStateMap
            tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct.leverageTo(ComputationGraph.workspaceTBPTT));
            tBpttStateMap.put(STATE_KEY_PREV_MEMCELL, fwdPass.lastMemCell.leverageTo(ComputationGraph.workspaceTBPTT));
        }

        return ActivationsFactory.getInstance().create(outAct, input.getMask(0), MaskState.Passthrough);
    }
}
