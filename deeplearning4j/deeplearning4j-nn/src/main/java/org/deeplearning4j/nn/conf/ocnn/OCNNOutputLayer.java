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

package org.deeplearning4j.nn.conf.ocnn;

import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseOutputLayer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.layers.ocnn.OCNNParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Collection;
import java.util.Map;

/**
 * An implementation of one class neural networks from:
 * <a href="https://arxiv.org/pdf/1802.06360.pdf">https://arxiv.org/pdf/1802.06360.pdf</a>
 *
 * The one class neural network approach is an extension of the standard output layer
 * with a single set of weights, an activation function, and a bias to:
 * 2 sets of weights, a learnable "r" parameter that is held static
 * 1 traditional set of weights.
 * 1 additional weight matrix
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
@JsonIgnoreProperties("lossFn")
public class OCNNOutputLayer extends BaseOutputLayer {
    //embedded hidden layer size
    //aka "K"
    private int hiddenSize;

    private double nu = 0.04;

    private int windowSize = 10000;

    private double initialRValue = 0.1;

    private boolean configureR = true;

    /**
     * Psuedo code from keras:
     *  start_time = time.time()
     for epoch in range(100):
     # Train with each example
     sess.run(updates, feed_dict={X: train_X,r:rvalue})
     rvalue = nnScore(train_X, w_1, w_2, g)
     with sess.as_default():
     rvalue = rvalue.eval()
     rvalue = np.percentile(rvalue,q=100*nu)
     print("Epoch = %d, r = %f"
     % (epoch + 1,rvalue))


     */
    private int lastEpochSinceRUpdated = 0;

    public OCNNOutputLayer(Builder builder) {
        super(builder);
        this.hiddenSize = builder.hiddenLayerSize;
        this.nu = builder.nu;
        this.activationFn = builder.activation;
        this.windowSize = builder.windowSize;
        this.initialRValue = builder.initialRValue;
        this.configureR = builder.configureR;

    }

    @JsonCreator
    @SuppressWarnings("unused")
    public OCNNOutputLayer(@JsonProperty("hiddenSize") int hiddenSize,@JsonProperty("nu") double nu,@JsonProperty("activation") IActivation activation,@JsonProperty("windowSize") int windowSize, @JsonProperty("initialRValue") double initialRValue,@JsonProperty("configureR") boolean configureR) {
        this.hiddenSize = hiddenSize;
        this.nu = nu;
        this.activationFn = activation;
        this.windowSize = windowSize;
        this.initialRValue = initialRValue;
        this.configureR = configureR;
    }

    @Override
    public ILossFunction getLossFn() {
        return lossFn;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("OCNNOutputLayer", getLayerName(), layerIndex, getNIn(), getNOut());

        org.deeplearning4j.nn.layers.ocnn.OCNNOutputLayer ret = new org.deeplearning4j.nn.layers.ocnn.OCNNOutputLayer(conf);
        ret.setListeners(trainingListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        ret.setActivation(activationFn);
        if(lastEpochSinceRUpdated == 0 && configureR)
            paramTable.get(OCNNParamInitializer.R_KEY).putScalar(0,initialRValue);
        return ret;
    }

    @Override
    public long getNOut() {
        //we don't change number of outputs here
        return 1L;
    }

    @Override
    public ParamInitializer initializer() {
        return OCNNParamInitializer.getInstance();
    }


    @Override
    public double getL1ByParam(String paramName) {
        return l1;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return l2;
    }

    @NoArgsConstructor
    public static class Builder extends BaseOutputLayer.Builder<Builder> {
        protected  int hiddenLayerSize;
        protected  double nu = 0.04;
        protected int windowSize = 10000;
        protected IActivation activation = new ActivationIdentity();
        protected  double initialRValue = 0.1;
        protected boolean configureR = true;

        /**
         * Whether to use the specified
         * {@link #initialRValue} or
         * use the weight initialization with
         * the neural network for the r value
         * @param configureR true if we should use the
         *                   initial {@link #initialRValue}
         *
         * @return
         */
        public Builder configureR(boolean configureR) {
            this.configureR = configureR;
            return this;
        }


        /**
         * The initial r value to use for ocnn
         * for definition, see the paper,
         * note this is only active when {@link #configureR}
         * is specified as true
         * @param initialRValue the int
         * @return
         */
        public Builder initialRValue(double initialRValue) {
            this.initialRValue = initialRValue;
            return this;
        }

        /**
         * The number of examples to use for computing the
         * quantile for the r value update.
         * This value should generally be the same
         * as the number of examples in the dataset
         * @param windowSize the number of examples to use
         *                   for computing the quantile
         *                   of the dataset for the r value update
         * @return
         */
        public Builder windowSize(int windowSize) {
            this.windowSize = windowSize;
            return this;
        }


        /**
         * For nu definition see the paper
         * @param nu the nu for ocnn
         * @return
         */
        public Builder nu(double nu) {
            this.nu = nu;
            return this;
        }

        /**
         * The activation function to use with ocnn
         * @param activation the activation function to sue
         * @return
         */
        public Builder activation(IActivation activation) {
            this.activation = activation;
            return this;
        }

        /**
         * The hidden layer size for the one class neural network.
         * Note this would be nOut on a dense layer.
         * NOut in this neural net is always set to 1 though.
         * @param hiddenLayerSize the hidden layer size to use
         *                        with ocnn
         * @return
         */
        public Builder hiddenLayerSize(int hiddenLayerSize) {
            this.hiddenLayerSize = hiddenLayerSize;
            return this;
        }

        @Override
        public Builder nOut(int nOut) {
            throw new UnsupportedOperationException("Unable to specify number of outputs with ocnn. Outputs are fixed to 1.");
        }

        @Override
        @SuppressWarnings("unchecked")
        public OCNNOutputLayer build() {
            return new OCNNOutputLayer(this);
        }
    }
}
