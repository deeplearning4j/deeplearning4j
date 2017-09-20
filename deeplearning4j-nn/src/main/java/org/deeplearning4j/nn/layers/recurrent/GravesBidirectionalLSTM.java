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
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.GravesBidirectionalLSTMParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 *
 * RNN tutorial: http://deeplearning4j.org/usingrnns.html
 * READ THIS FIRST
 *
 * Bdirectional LSTM layer implementation.
 * Based on Graves: Supervised Sequence Labelling with Recurrent Neural Networks
 * http://www.cs.toronto.edu/~graves/phd.pdf
 * See also for full/vectorized equations (and a comparison to other LSTM variants):
 * Greff et al. 2015, "LSTM: A Search Space Odyssey", pg11. This is the "vanilla" variant in said paper
 * http://arxiv.org/pdf/1503.04069.pdf
 *
 * A high level description of bidirectional LSTM can be found from
 * "Hybrid Speech Recognition with Deep Bidirectional LSTM"
 *  http://www.cs.toronto.edu/~graves/asru_2013.pdf
 *
 *
 * @author Alex Black
 * @author Benjamin Joseph
 */
@Slf4j
public class GravesBidirectionalLSTM
                extends BaseRecurrentLayer<org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM> {

    protected FwdPassReturn cachedPassForward;
    protected FwdPassReturn cachedPassBackward;

    public GravesBidirectionalLSTM(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Gradient gradient() {
        throw new UnsupportedOperationException("Not supported " + layerId());
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

        if (truncatedBPTT) {
            throw new UnsupportedOperationException(
                            "Time step for bidirectional RNN not supported: it has to run on a batch of data all at once "
                                            + layerId());
        }

        final FwdPassReturn fwdPass = activateHelperDirectional(true, null, null, true, true);

        final Gradients forwardsGradient = LSTMHelpers.backpropGradientHelper(this.conf,
                        this.layerConf().getGateActivationFn(), this.input.get(0),
                        getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS),
                        getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS), epsilon,
                        truncatedBPTT, tbpttBackwardLength, fwdPass, true,
                        GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS,
                        GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS,
                        GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS, gradientViews, this.input.getMask(0), true,
                        null);



        final FwdPassReturn backPass = activateHelperDirectional(true, null, null, true, false);

        final Gradients backwardsGradient = LSTMHelpers.backpropGradientHelper(this.conf,
                        this.layerConf().getGateActivationFn(), this.input.get(0),
                        getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS),
                        getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS), epsilon,
                        truncatedBPTT, tbpttBackwardLength, backPass, false,
                        GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS,
                        GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS,
                        GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS, gradientViews, this.input.getMask(0), true,
                        null);


        //merge the gradient, which is key value pair of String,INDArray
        //the keys for forwards and backwards should be different

        final Gradient combinedGradient = new DefaultGradient();


        for (Map.Entry<String, INDArray> entry : forwardsGradient.getParameterGradients().gradientForVariable().entrySet()) {
            combinedGradient.setGradientFor(entry.getKey(), entry.getValue());
        }

        for (Map.Entry<String, INDArray> entry : backwardsGradient.getParameterGradients().gradientForVariable().entrySet()) {
            combinedGradient.setGradientFor(entry.getKey(), entry.getValue());
        }

        final Gradient correctOrderedGradient = new DefaultGradient();

        for (final String key : params.keySet()) {
            correctOrderedGradient.setGradientFor(key, combinedGradient.getGradientFor(key));
        }

        final INDArray forwardEpsilon = forwardsGradient.get(0);
        final INDArray backwardsEpsilon = backwardsGradient.get(0);
        final INDArray combinedEpsilon = forwardEpsilon.addi(backwardsEpsilon);

        //sum the errors that were back-propagated
        Gradients g = GradientsFactory.getInstance().create(combinedEpsilon, correctOrderedGradient);
        return backpropPreprocessor(g);
    }

    @Override
    public Activations activate(Activations input, boolean training) {
        setInput(input);
        return activateOutput(training, false);
    }

    @Override
    public Activations activate(boolean training) {
        return activateOutput(training, false);
    }

    private Activations activateOutput(final boolean training, boolean forBackprop) {
        final FwdPassReturn forwardsEval;
        final FwdPassReturn backwardsEval;

        if (cacheMode != CacheMode.NONE && cachedPassForward != null && cachedPassBackward != null) {
            // restore from cache. but this coll will probably never happen
            forwardsEval = cachedPassForward;
            backwardsEval = cachedPassBackward;

            cachedPassBackward = null;
            cachedPassForward = null;
        } else {

            forwardsEval = LSTMHelpers.activateHelper(this, this.conf, this.layerConf().getGateActivationFn(),
                            this.input.get(0), getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS),
                            getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS),
                            getParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS), training, null, null,
                            forBackprop || (cacheMode != CacheMode.NONE && training), true,
                            GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS, this.input.getMask(0), true, null,
                            forBackprop ? cacheMode : CacheMode.NONE);

            backwardsEval = LSTMHelpers.activateHelper(this, this.conf, this.layerConf().getGateActivationFn(),
                            this.input.get(0),
                            getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS),
                            getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS),
                            getParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS), training, null, null,
                            forBackprop || (cacheMode != CacheMode.NONE && training), false,
                            GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS, this.input.getMask(0), true, null,
                            forBackprop ? cacheMode : CacheMode.NONE);

            cachedPassForward = forwardsEval;
            cachedPassBackward = backwardsEval;
        }

        //sum outputs
        final INDArray fwdOutput = forwardsEval.fwdPassOutput;
        final INDArray backOutput = backwardsEval.fwdPassOutput;

        // if we're on ff pass & cache enabled - we should not modify fwdOutput, and for backprop pass - we don't care
        final INDArray totalOutput = training && cacheMode != CacheMode.NONE && !forBackprop ? fwdOutput.add(backOutput)
                        : fwdOutput.addi(backOutput);

        //Bidirectional RNNs operate differently to standard RNNs from a masking perspective
        //Specifically, the masks are applied regardless of the mask state
        //For example, input -> RNN -> Bidirectional-RNN: we should still mask the activations and errors in the bi-RNN
        // even though the normal RNN has marked the current mask state as 'passthrough'
        //Consequently, the mask is marked as active again

        return ActivationsFactory.getInstance().create(totalOutput, input.getMask(0), (input.getMask(0) == null ? null : MaskState.Active));
    }

    private FwdPassReturn activateHelperDirectional(final boolean training, final INDArray prevOutputActivations,
                    final INDArray prevMemCellState, boolean forBackprop, boolean forwards) {

        if (cacheMode == null)
            cacheMode = CacheMode.NONE;

        if (cacheMode != CacheMode.NONE && forwards && forBackprop && cachedPassForward != null) {
            FwdPassReturn ret = cachedPassForward;
            cachedPassForward = null;
            return ret;
        } else if (cacheMode != CacheMode.NONE && !forwards && forBackprop) {
            FwdPassReturn ret = cachedPassBackward;
            cachedPassBackward = null;
            return ret;
        } else {

            String recurrentKey = GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS;
            String inputKey = GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS;
            String biasKey = GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS;

            if (!forwards) {
                recurrentKey = GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS;
                inputKey = GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS;
                biasKey = GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS;
            }

            return LSTMHelpers.activateHelper(this, this.conf, this.layerConf().getGateActivationFn(), this.input.get(0),
                            getParam(recurrentKey), getParam(inputKey), getParam(biasKey), training,
                            prevOutputActivations, prevMemCellState, forBackprop, forwards, inputKey, this.input.getMask(0), true,
                            null, forBackprop ? cacheMode : CacheMode.NONE);
        }
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public Activations rnnTimeStep(Activations input) {
        throw new UnsupportedOperationException(
                        "you can not time step a bidirectional RNN, it has to run on a batch of data all at once "
                                        + layerId());
    }



    @Override
    public INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT) {
        throw new UnsupportedOperationException(
                        "Cannot set stored state: bidirectional RNNs don't have stored state " + layerId());
    }
}
