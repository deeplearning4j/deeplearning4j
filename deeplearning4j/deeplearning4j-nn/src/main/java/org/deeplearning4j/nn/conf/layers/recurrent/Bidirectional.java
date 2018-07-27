/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.conf.layers.recurrent;

import lombok.*;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.api.layers.RecurrentLayer;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.layers.recurrent.BidirectionalLayer;
import org.deeplearning4j.nn.params.BidirectionalParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;

import java.util.Collection;
import java.util.Map;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * Bidirectional is a "wrapper" layer: it wraps any uni-directional RNN layer to make it bidirectional.<br>
 * Note that multiple different modes are supported - these specify how the activations should be combined from
 * the forward and backward RNN networks. See {@link Mode} javadoc for more details.<br>
 * Parameters are not shared here - there are 2 separate copies of the wrapped RNN layer, each with separate parameters.
 * <br>
 * Usage: {@code .layer(new Bidirectional(new LSTM.Builder()....build())}
 *
 * @author Alex Black
 */
@NoArgsConstructor
@Data
@EqualsAndHashCode(callSuper = true, exclude = {"initializer"})
@JsonIgnoreProperties({"initializer"})
public class Bidirectional extends Layer {

    /**
     * This Mode enumeration defines how the activations for the forward and backward networks should be combined.<br>
     * ADD: out = forward + backward (elementwise addition)<br>
     * MUL: out = forward * backward (elementwise multiplication)<br>
     * AVERAGE: out = 0.5 * (forward + backward)<br>
     * CONCAT: Concatenate the activations.<br>
     * Where 'forward' is the activations for the forward RNN, and 'backward' is the activations for the backward RNN.
     * In all cases except CONCAT, the output activations size is the same size as the standard RNN that is being wrapped
     * by this layer. In the CONCAT case, the output activations size (dimension 1) is 2x larger than the standard RNN's
     * activations array.
     */
    public enum Mode {
        ADD, MUL, AVERAGE, CONCAT
    }

    private Layer fwd;
    private Layer bwd;
    private Mode mode;
    private transient BidirectionalParamInitializer initializer;

    private Bidirectional(Bidirectional.Builder builder) {
        super(builder);
    }

    /**
     * Create a Bidirectional wrapper, with the default Mode (CONCAT) for the specified layer
     *
     * @param layer layer to wrap
     */
    public Bidirectional(@NonNull Layer layer) {
        this(Mode.CONCAT, layer);
    }

    /**
     * Create a Bidirectional wrapper for the specified layer
     *
     * @param mode  Mode to use to combine activations. See {@link Mode} for details
     * @param layer layer to wrap
     */
    public Bidirectional(@NonNull Mode mode, @NonNull Layer layer) {
        if (!(layer instanceof BaseRecurrentLayer || layer instanceof LastTimeStep || layer instanceof BaseWrapperLayer)) {
            throw new IllegalArgumentException("Cannot wrap a non-recurrent layer: " +
                    "config must extend BaseRecurrentLayer or LastTimeStep " +
                    "Got class: " + layer.getClass());
        }
        this.fwd = layer;
        this.bwd = layer.clone();
        this.mode = mode;
    }

    public long getNOut() {
        if (this.fwd instanceof LastTimeStep) {
            return ((FeedForwardLayer) ((LastTimeStep) this.fwd).getUnderlying()).getNOut();
        } else {
            return ((FeedForwardLayer) this.fwd).getNOut();
        }
    }

    public long getNIn() {
        if (this.fwd instanceof LastTimeStep) {
            return  ((FeedForwardLayer)((LastTimeStep) this.fwd).getUnderlying()).getNIn();
        } else {
            return ((FeedForwardLayer) this.fwd).getNIn();
        }
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> trainingListeners, int layerIndex,
                                                       INDArray layerParamsView, boolean initializeParams) {
        NeuralNetConfiguration c1 = conf.clone();
        NeuralNetConfiguration c2 = conf.clone();
        c1.setLayer(fwd);
        c2.setLayer(bwd);

        long n = layerParamsView.length() / 2;
        INDArray fp = layerParamsView.get(point(0), interval(0, n));
        INDArray bp = layerParamsView.get(point(0), interval(n, 2 * n));
        org.deeplearning4j.nn.api.Layer f
                = fwd.instantiate(c1, trainingListeners, layerIndex, fp, initializeParams);

        org.deeplearning4j.nn.api.Layer b
                = bwd.instantiate(c2, trainingListeners, layerIndex, bp, initializeParams);

        BidirectionalLayer ret = new BidirectionalLayer(conf, f, b, layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);

        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        if (initializer == null) {
            initializer = new BidirectionalParamInitializer(this);
        }
        return initializer;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        InputType outOrig = fwd.getOutputType(layerIndex, inputType);

        if (fwd instanceof LastTimeStep) {
            InputType.InputTypeFeedForward ff = (InputType.InputTypeFeedForward) outOrig;
            if (mode == Mode.CONCAT) {
                return InputType.feedForward(2 * ff.getSize());
            } else {
                return ff;
            }
        } else {
            InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) outOrig;
            if (mode == Mode.CONCAT) {
                return InputType.recurrent(2 * r.getSize());
            } else {
                return r;
            }
        }
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        fwd.setNIn(inputType, override);
        bwd.setNIn(inputType, override);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return fwd.getPreProcessorForInputType(inputType);
    }

    @Override
    public boolean isPretrain() {
        return fwd.isPretrain();
    }

    @Override
    public double getL1ByParam(String paramName) {
        //Strip forward/backward prefix from param name
        return fwd.getL1ByParam(paramName.substring(1));
    }

    @Override
    public double getL2ByParam(String paramName) {
        return fwd.getL2ByParam(paramName.substring(1));
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return fwd.isPretrainParam(paramName.substring(1));
    }

    /**
     * Get the updater for the given parameter. Typically the same updater will be used for all updaters, but this
     * is not necessarily the case
     *
     * @param paramName Parameter name
     * @return IUpdater for the parameter
     */
    public IUpdater getUpdaterByParam(String paramName) {
        String sub = paramName.substring(1);
        return fwd.getUpdaterByParam(sub);
    }

    @Override
    public GradientNormalization getGradientNormalization() {
        return fwd.getGradientNormalization();
    }

    @Override
    public double getGradientNormalizationThreshold() {
        return fwd.getGradientNormalizationThreshold();
    }

    @Override
    public void setLayerName(String layerName) {
        this.layerName = layerName;
        fwd.setLayerName(layerName);
        bwd.setLayerName(layerName);
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        LayerMemoryReport lmr = fwd.getMemoryReport(inputType);
        lmr.scale(2);   //Double all memory use
        return lmr;
    }

    @AllArgsConstructor
    public static class Builder extends Layer.Builder<Bidirectional.Builder> {

        private Mode mode;
        private Layer layer;

        public Builder mode(Mode mode) {
            this.mode = mode;
            return this;
        }

        public Builder rnnLayer(Layer layer) {
            if (!(layer instanceof BaseRecurrentLayer || layer instanceof LastTimeStep
                    || layer instanceof BaseWrapperLayer)) {
                throw new IllegalArgumentException("Cannot wrap a non-recurrent layer: " +
                        "config must extend BaseRecurrentLayer or LastTimeStep " +
                        "Got class: " + layer.getClass());
            }
            this.layer = layer;
            return this;
        }

        @SuppressWarnings("unchecked")
        public Bidirectional build() {
            return new Bidirectional(this);
        }
    }
}
