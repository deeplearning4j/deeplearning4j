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

package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.CenterLossParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.Collection;
import java.util.Map;

/**
 * Center loss is similar to triplet loss except that it enforces
 * intraclass consistency and doesn't require feed forward of multiple
 * examples. Center loss typically converges faster for training
 * ImageNet-based convolutional networks.
 *
 * "If example x is in class Y, ensure that embedding(x) is close to
 * {@code average(embedding(y))} for all examples y in Y"
 *
 * @author Justin Long (@crockpotveggies)
 * @author Alex Black (@AlexDBlack)
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class CenterLossOutputLayer extends BaseOutputLayer {
    protected double alpha;
    protected double lambda;
    protected boolean gradientCheck;

    protected CenterLossOutputLayer(Builder builder) {
        super(builder);
        this.alpha = builder.alpha;
        this.lambda = builder.lambda;
        this.gradientCheck = builder.gradientCheck;
        initializeConstraints(builder);
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
                    int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("CenterLossOutputLayer", getLayerName(), layerIndex, getNIn(), getNOut());

        Layer ret = new org.deeplearning4j.nn.layers.training.CenterLossOutputLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return CenterLossParamInitializer.getInstance();
    }

    @Override
    public IUpdater getUpdaterByParam(String paramName) {
        // center loss utilizes alpha directly for this so any updater can be used for other layers
        switch (paramName) {
            case CenterLossParamInitializer.CENTER_KEY:
                return new NoOp();
            default:
                return iUpdater;
        }
    }

    @Override
    public double getL1ByParam(String paramName) {
        switch (paramName) {
            case CenterLossParamInitializer.WEIGHT_KEY:
                return l1;
            case CenterLossParamInitializer.BIAS_KEY:
                return l1Bias;
            case CenterLossParamInitializer.CENTER_KEY:
                return 0.0;
            default:
                throw new IllegalStateException("Unknown parameter: \"" + paramName + "\"");
        }
    }

    @Override
    public double getL2ByParam(String paramName) {
        switch (paramName) {
            case CenterLossParamInitializer.WEIGHT_KEY:
                return l2;
            case CenterLossParamInitializer.BIAS_KEY:
                return l2Bias;
            case CenterLossParamInitializer.CENTER_KEY:
                return 0.0;
            default:
                throw new IllegalStateException("Unknown parameter: \"" + paramName + "\"");
        }
    }

    public double getAlpha() {
        return alpha;
    }

    public double getLambda() {
        return lambda;
    }

    public boolean getGradientCheck() {
        return gradientCheck;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //Basically a dense layer, with some extra params...
        InputType outputType = getOutputType(-1, inputType);

        val nParamsW = nIn * nOut;
        val nParamsB = nOut;
        val nParamsCenter = nIn * nOut;
        val numParams = nParamsW + nParamsB + nParamsCenter;

        int updaterStateSize = (int) (getUpdaterByParam(CenterLossParamInitializer.WEIGHT_KEY).stateSize(nParamsW)
                        + getUpdaterByParam(CenterLossParamInitializer.BIAS_KEY).stateSize(nParamsB)
                        + getUpdaterByParam(CenterLossParamInitializer.CENTER_KEY).stateSize(nParamsCenter));

        int trainSizeFixed = 0;
        int trainSizeVariable = 0;
        if (getIDropout() != null) {
            if (false) {
                //TODO drop connect
                //Dup the weights... note that this does NOT depend on the minibatch size...
                trainSizeVariable += 0; //TODO
            } else {
                //Assume we dup the input
                trainSizeVariable += inputType.arrayElementsPerExample();
            }
        }

        //Also, during backprop: we do a preOut call -> gives us activations size equal to the output size
        // which is modified in-place by activation function backprop
        // then we have 'epsilonNext' which is equivalent to input size
        trainSizeVariable += outputType.arrayElementsPerExample();

        return new LayerMemoryReport.Builder(layerName, CenterLossOutputLayer.class, inputType, outputType)
                        .standardMemory(numParams, updaterStateSize)
                        .workingMemory(0, 0, trainSizeFixed, trainSizeVariable) //No additional memory (beyond activations) for inference
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                        .build();
    }

    @NoArgsConstructor
    public static class Builder extends BaseOutputLayer.Builder<Builder> {
        protected double alpha = 0.05;
        protected double lambda = 2e-4;
        protected boolean gradientCheck = false;

        public Builder(LossFunction lossFunction) {
            super.lossFunction(lossFunction);
        }

        public Builder(ILossFunction lossFunction) {
            this.lossFn = lossFunction;
        }

        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public Builder lambda(double lambda) {
            this.lambda = lambda;
            return this;
        }

        public Builder gradientCheck(boolean isGradientCheck) {
            this.gradientCheck = isGradientCheck;
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")
        public CenterLossOutputLayer build() {
            return new CenterLossOutputLayer(this);
        }
    }
}

