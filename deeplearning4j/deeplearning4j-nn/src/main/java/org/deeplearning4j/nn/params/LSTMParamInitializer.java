/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.params;

import lombok.val;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

public class LSTMParamInitializer implements ParamInitializer {

    private static final LSTMParamInitializer INSTANCE = new LSTMParamInitializer();

    public static LSTMParamInitializer getInstance() {
        return INSTANCE;
    }

    /** Weights for previous time step -> current time step connections */
    public final static String RECURRENT_WEIGHT_KEY = "RW";
    public final static String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;
    public final static String INPUT_WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;

    private static final List<String> LAYER_PARAM_KEYS = Collections.unmodifiableList(
            Arrays.asList(INPUT_WEIGHT_KEY, RECURRENT_WEIGHT_KEY, BIAS_KEY));
    private static final List<String> WEIGHT_KEYS = Collections.unmodifiableList(
            Arrays.asList(INPUT_WEIGHT_KEY, RECURRENT_WEIGHT_KEY));
    private static final List<String> BIAS_KEYS = Collections.unmodifiableList(Collections.singletonList(BIAS_KEY));

    @Override
    public long numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public long numParams(Layer l) {
        LSTM layerConf = (LSTM) l;

        val nL = layerConf.getNOut(); //i.e., n neurons in this layer
        val nLast = layerConf.getNIn(); //i.e., n neurons in previous layer

        val nParams = nLast * (4 * nL) //"input" weights
                        + nL * (4 * nL) //recurrent weights
                        + 4 * nL; //bias

        return nParams;
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        return LAYER_PARAM_KEYS;
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return WEIGHT_KEYS;
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        return BIAS_KEYS;
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return RECURRENT_WEIGHT_KEY.equals(key) || INPUT_WEIGHT_KEY.equals(key);
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return BIAS_KEY.equals(key);
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());
        LSTM layerConf = (LSTM) conf.getLayer();
        double forgetGateInit = layerConf.getForgetGateBiasInit();

        val nL = layerConf.getNOut(); //i.e., n neurons in this layer
        val nLast = layerConf.getNIn(); //i.e., n neurons in previous layer

        conf.addVariable(INPUT_WEIGHT_KEY);
        conf.addVariable(RECURRENT_WEIGHT_KEY);
        conf.addVariable(BIAS_KEY);

        val length = numParams(conf);
        if (paramsView.length() != length)
            throw new IllegalStateException(
                            "Expected params view of length " + length + ", got length " + paramsView.length());

        INDArray paramsViewReshape = paramsView.reshape(paramsView.length());
        val nParamsIn = nLast * (4 * nL);
        val nParamsRecurrent = nL * (4 * nL);
        val nBias = 4 * nL;
        INDArray inputWeightView = paramsViewReshape.get( NDArrayIndex.interval(0, nParamsIn));
        INDArray recurrentWeightView = paramsViewReshape.get(
                        NDArrayIndex.interval(nParamsIn, nParamsIn + nParamsRecurrent));
        INDArray biasView = paramsViewReshape.get(
                        NDArrayIndex.interval(nParamsIn + nParamsRecurrent, nParamsIn + nParamsRecurrent + nBias));

        if (initializeParams) {
            val fanIn = nL;
            val fanOut = nLast + nL;
            val inputWShape = new long[] {nLast, 4 * nL};
            val recurrentWShape = new long[] {nL, 4 * nL};

            IWeightInit rwInit;
            if(layerConf.getWeightInitFnRecurrent() != null){
                rwInit = layerConf.getWeightInitFnRecurrent();
            } else {
                rwInit = layerConf.getWeightInitFn();
            }

            params.put(INPUT_WEIGHT_KEY, layerConf.getWeightInitFn().init(fanIn, fanOut, inputWShape,
                    IWeightInit.DEFAULT_WEIGHT_INIT_ORDER, inputWeightView));
            INDArray init = rwInit.init(fanIn, fanOut, recurrentWShape, IWeightInit.DEFAULT_WEIGHT_INIT_ORDER, recurrentWeightView);
            params.put(RECURRENT_WEIGHT_KEY, init);
            biasView.put(new INDArrayIndex[] {NDArrayIndex.interval(nL, 2 * nL)},
                            Nd4j.valueArrayOf(new long[]{nL}, forgetGateInit)); //Order: input, forget, output, input modulation, i.e., IFOG}
            /*The above line initializes the forget gate biases to specified value.
             * See Sutskever PhD thesis, pg19:
             * "it is important for [the forget gate activations] to be approximately 1 at the early stages of learning,
             *  which is accomplished by initializing [the forget gate biases] to a large value (such as 5). If it is
             *  not done, it will be harder to learn long range dependencies because the smaller values of the forget
             *  gates will create a vanishing gradients problem."
             *  http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
             */
            params.put(BIAS_KEY, biasView);
        } else {
            params.put(INPUT_WEIGHT_KEY, WeightInitUtil.reshapeWeights(new long[] {nLast, 4 * nL}, inputWeightView));
            params.put(RECURRENT_WEIGHT_KEY,
                            WeightInitUtil.reshapeWeights(new long[] {nL, 4 * nL}, recurrentWeightView));
            params.put(BIAS_KEY, biasView);
        }

        return params;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        LSTM layerConf = (LSTM) conf.getLayer();

        val nL = layerConf.getNOut(); //i.e., n neurons in this layer
        val nLast = layerConf.getNIn(); //i.e., n neurons in previous layer

        val length = numParams(conf);
        if (gradientView.length() != length)
            throw new IllegalStateException(
                            "Expected gradient view of length " + length + ", got length " + gradientView.length());

        val nParamsIn = nLast * (4 * nL);
        val nParamsRecurrent = nL * (4 * nL);
        val nBias = 4 * nL;
        INDArray gradientViewReshape = gradientView.reshape(gradientView.length());
        INDArray inputWeightGradView = gradientViewReshape.get( NDArrayIndex.interval(0, nParamsIn))
                        .reshape('f', nLast, 4 * nL);
        INDArray recurrentWeightGradView = gradientViewReshape
                        .get(NDArrayIndex.interval(nParamsIn, nParamsIn + nParamsRecurrent))
                        .reshape('f', nL, 4 * nL);
        INDArray biasGradView = gradientViewReshape.get(
                        NDArrayIndex.interval(nParamsIn + nParamsRecurrent, nParamsIn + nParamsRecurrent + nBias)); //already a row vector

        Map<String, INDArray> out = new LinkedHashMap<>();
        out.put(INPUT_WEIGHT_KEY, inputWeightGradView);
        out.put(RECURRENT_WEIGHT_KEY, recurrentWeightGradView);
        out.put(BIAS_KEY, biasGradView);

        return out;
    }
}
