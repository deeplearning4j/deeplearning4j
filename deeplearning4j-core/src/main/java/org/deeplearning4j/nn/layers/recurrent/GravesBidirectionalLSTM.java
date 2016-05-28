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
import org.deeplearning4j.nn.params.GravesBidirectionalLSTMParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

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
 * PLEASE NOTE that truncated backpropagation through time (BPTT) will not work with the bidirectional layer as-is.
 * Additionally, variable length data sets will also not work with the bidirectional layer.
 *
 * @author Alex Black
 * @author Benjamin Joseph
 */
public class GravesBidirectionalLSTM extends BaseRecurrentLayer<org.deeplearning4j.nn.conf.layers.GravesLSTM> {

    public GravesBidirectionalLSTM(NeuralNetConfiguration conf) {
        super(conf);
    }

    public GravesBidirectionalLSTM(NeuralNetConfiguration conf, INDArray input) {
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

        if (truncatedBPTT) {
            throw new UnsupportedOperationException("you can not time step a bidirectional RNN, it has to run on a batch of data all at once");
        }

        final FwdPassReturn fwdPass = activateHelperDirectional(true, null, null, true,true);

        final Pair<Gradient, INDArray> forwardsGradient = LSTMHelpers.backpropGradientHelper(
                this.conf,
                this.input,
                getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS),
                getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS),
                epsilon,
                truncatedBPTT,
                tbpttBackwardLength,
                fwdPass,
                true,
                GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS,
                GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS,
                GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS,
                gradientViews);



        final FwdPassReturn backPass = activateHelperDirectional(true, null, null, true,false);

        final Pair<Gradient, INDArray> backwardsGradient = LSTMHelpers.backpropGradientHelper(
                this.conf,
                this.input,
                getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS),
                getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS),
                epsilon,
                truncatedBPTT,
                tbpttBackwardLength,
                backPass,
                false,
                GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS,
                GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS,
                GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS,
                gradientViews);


        //merge the gradient, which is key value pair of String,INDArray
        //the keys for forwards and backwards should be different

        final Gradient combinedGradient = new DefaultGradient();


        for (Map.Entry<String,INDArray> entry : forwardsGradient.getFirst().gradientForVariable().entrySet()) {
            combinedGradient.setGradientFor(entry.getKey(),entry.getValue());
        }

        for (Map.Entry<String,INDArray> entry : backwardsGradient.getFirst().gradientForVariable().entrySet()) {
            combinedGradient.setGradientFor(entry.getKey(),entry.getValue());
        }

        final Gradient correctOrderedGradient = new DefaultGradient();

        for (final String key : params.keySet()) {
            correctOrderedGradient.setGradientFor(key,combinedGradient.getGradientFor(key));
        }

        final INDArray forwardEpsilon = forwardsGradient.getSecond();
        final INDArray backwardsEpsilon = backwardsGradient.getSecond();
        final INDArray combinedEpsilon = forwardEpsilon.addi(backwardsEpsilon);

        //sum the errors that were back-propagated
        return  new Pair<>(correctOrderedGradient,combinedEpsilon );

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
        return activateOutput(training, false);
    }

    @Override
    public INDArray activate(INDArray input) {
        setInput(input);
        return activateOutput(true, false);
    }

    @Override
    public INDArray activate(boolean training) {
        return activateOutput(training, false);
    }

    @Override
    public INDArray activate() {

        return activateOutput(false, false);
    }

    private INDArray activateOutput(final boolean training, boolean forBackprop) {


        final FwdPassReturn forwardsEval = LSTMHelpers.activateHelper(
                this,
                this.conf,
                this.input,
                getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS),
                getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS),
                getParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS),
                training,null,null,forBackprop,true,
                GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS);

        final FwdPassReturn backwardsEval = LSTMHelpers.activateHelper(
                this,
                this.conf,
                this.input,
                getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS),
                getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS),
                getParam(GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS),
                training,null,null,forBackprop,false,
                GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS);


        //sum outputs
        final INDArray fwdOutput = forwardsEval.fwdPassOutput;
        final INDArray backOutput = backwardsEval.fwdPassOutput;
        final INDArray totalOutput = fwdOutput.addi(backOutput);

        return totalOutput;
    }

    private FwdPassReturn activateHelperDirectional(final boolean training,
                                         final INDArray prevOutputActivations,
                                         final INDArray prevMemCellState,
                                         boolean forBackprop,
                                         boolean forwards) {

        String recurrentKey = GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS;
        String inputKey = GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS;
        String biasKey = GravesBidirectionalLSTMParamInitializer.BIAS_KEY_FORWARDS;

        if (!forwards) {
            recurrentKey = GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS;
            inputKey = GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS;
            biasKey = GravesBidirectionalLSTMParamInitializer.BIAS_KEY_BACKWARDS;
        }

        return LSTMHelpers.activateHelper(
                this,
                this.conf,
                this.input,
                getParam(recurrentKey),
                getParam(inputKey),
                getParam(biasKey),
                training,
                prevOutputActivations,
                prevMemCellState,
                forBackprop,
                forwards,
                inputKey);

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
        double l2Norm = getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS).norm2Number().doubleValue();
        double sumSquaredWeights = l2Norm*l2Norm;

        l2Norm = getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS).norm2Number().doubleValue();
        sumSquaredWeights += l2Norm*l2Norm;

        l2Norm = getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS).norm2Number().doubleValue();
        sumSquaredWeights += l2Norm*l2Norm;

        l2Norm = getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS).norm2Number().doubleValue();
        sumSquaredWeights += l2Norm * l2Norm;

        return 0.5 * conf.getLayer().getL2() * sumSquaredWeights;
    }



    @Override
    public double calcL1() {
        if (!conf.isUseRegularization() || conf.getLayer().getL1() <= 0.0) return 0.0;
        double l1 = getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_FORWARDS).norm1Number().doubleValue()
                + getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_FORWARDS).norm1Number().doubleValue()
                + getParam(GravesBidirectionalLSTMParamInitializer.RECURRENT_WEIGHT_KEY_BACKWARDS).norm1Number().doubleValue()
                + getParam(GravesBidirectionalLSTMParamInitializer.INPUT_WEIGHT_KEY_BACKWARDS).norm1Number().doubleValue();

        return conf.getLayer().getL1() * l1;
    }

    @Override
    public INDArray rnnTimeStep(INDArray input) {
        throw new UnsupportedOperationException("you can not time step a bidirectional RNN, it has to run on a batch of data all at once");
    }



    @Override
    public INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT) {
        throw new UnsupportedOperationException("no such thing as stored state for bidirectional RNN");
    }
}
