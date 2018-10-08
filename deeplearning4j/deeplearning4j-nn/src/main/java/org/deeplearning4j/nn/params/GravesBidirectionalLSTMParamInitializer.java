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

package org.deeplearning4j.nn.params;

import lombok.val;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

/**LSTM Parameter initializer, for LSTM based on
 * Graves: Supervised Sequence Labelling with Recurrent Neural Networks
 * <a href="http://www.cs.toronto.edu/~graves/phd.pdf">http://www.cs.toronto.edu/~graves/phd.pdf</a>
 */
public class GravesBidirectionalLSTMParamInitializer implements ParamInitializer {

    private static final GravesBidirectionalLSTMParamInitializer INSTANCE =
                    new GravesBidirectionalLSTMParamInitializer();

    public static GravesBidirectionalLSTMParamInitializer getInstance() {
        return INSTANCE;
    }

    /** Weights for previous time step -> current time step connections */
    public final static String RECURRENT_WEIGHT_KEY_FORWARDS = "RWF";
    public final static String BIAS_KEY_FORWARDS = DefaultParamInitializer.BIAS_KEY + "F";
    public final static String INPUT_WEIGHT_KEY_FORWARDS = DefaultParamInitializer.WEIGHT_KEY + "F";

    public final static String RECURRENT_WEIGHT_KEY_BACKWARDS = "RWB";
    public final static String BIAS_KEY_BACKWARDS = DefaultParamInitializer.BIAS_KEY + "B";
    public final static String INPUT_WEIGHT_KEY_BACKWARDS = DefaultParamInitializer.WEIGHT_KEY + "B";

    private static final List<String> WEIGHT_KEYS = Collections.unmodifiableList(Arrays.asList(INPUT_WEIGHT_KEY_FORWARDS,
            INPUT_WEIGHT_KEY_BACKWARDS, RECURRENT_WEIGHT_KEY_FORWARDS, RECURRENT_WEIGHT_KEY_BACKWARDS));
    private static final List<String> BIAS_KEYS = Collections.unmodifiableList(Arrays.asList(BIAS_KEY_FORWARDS, BIAS_KEY_BACKWARDS));
    private static final List<String> ALL_PARAM_KEYS = Collections.unmodifiableList(Arrays.asList(INPUT_WEIGHT_KEY_FORWARDS,
            INPUT_WEIGHT_KEY_BACKWARDS, RECURRENT_WEIGHT_KEY_FORWARDS, RECURRENT_WEIGHT_KEY_BACKWARDS, BIAS_KEY_FORWARDS,
            BIAS_KEY_BACKWARDS));

    @Override
    public long numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public long numParams(Layer l) {
        org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM layerConf =
                        (org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM) l;

        val nL = layerConf.getNOut(); //i.e., n neurons in this layer
        val nLast = layerConf.getNIn(); //i.e., n neurons in previous layer

        val nParamsForward = nLast * (4 * nL) //"input" weights
                        + nL * (4 * nL + 3) //recurrent weights
                        + 4 * nL; //bias

        return 2 * nParamsForward;
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        return ALL_PARAM_KEYS;
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
        return RECURRENT_WEIGHT_KEY_FORWARDS.equals(key) || INPUT_WEIGHT_KEY_FORWARDS.equals(key)
                || RECURRENT_WEIGHT_KEY_BACKWARDS.equals(key) || INPUT_WEIGHT_KEY_BACKWARDS.equals(key);
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return BIAS_KEY_FORWARDS.equals(key) || BIAS_KEY_BACKWARDS.equals(key);
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());

        org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM layerConf =
                        (org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM) conf.getLayer();
        double forgetGateInit = layerConf.getForgetGateBiasInit();

        Distribution dist = Distributions.createDistribution(layerConf.getDist());

        val nL = layerConf.getNOut(); //i.e., n neurons in this layer
        val nLast = layerConf.getNIn(); //i.e., n neurons in previous layer

        conf.addVariable(INPUT_WEIGHT_KEY_FORWARDS);
        conf.addVariable(RECURRENT_WEIGHT_KEY_FORWARDS);
        conf.addVariable(BIAS_KEY_FORWARDS);
        conf.addVariable(INPUT_WEIGHT_KEY_BACKWARDS);
        conf.addVariable(RECURRENT_WEIGHT_KEY_BACKWARDS);
        conf.addVariable(BIAS_KEY_BACKWARDS);

        val nParamsInput = nLast * (4 * nL);
        val nParamsRecurrent = nL * (4 * nL + 3);
        val nBias = 4 * nL;

        val rwFOffset = nParamsInput;
        val bFOffset = rwFOffset + nParamsRecurrent;
        val iwROffset = bFOffset + nBias;
        val rwROffset = iwROffset + nParamsInput;
        val bROffset = rwROffset + nParamsRecurrent;

        INDArray iwF = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, rwFOffset));
        INDArray rwF = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(rwFOffset, bFOffset));
        INDArray bF = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(bFOffset, iwROffset));
        INDArray iwR = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(iwROffset, rwROffset));
        INDArray rwR = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(rwROffset, bROffset));
        INDArray bR = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(bROffset, bROffset + nBias));

        if (initializeParams) {
            bF.put(new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.interval(nL, 2 * nL)},
                            Nd4j.ones(1, nL).muli(forgetGateInit)); //Order: input, forget, output, input modulation, i.e., IFOG
            bR.put(new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.interval(nL, 2 * nL)},
                            Nd4j.ones(1, nL).muli(forgetGateInit));
        }
        /*The above line initializes the forget gate biases to specified value.
         * See Sutskever PhD thesis, pg19:
         * "it is important for [the forget gate activations] to be approximately 1 at the early stages of learning,
         *  which is accomplished by initializing [the forget gate biases] to a large value (such as 5). If it is
         *  not done, it will be harder to learn long range dependencies because the smaller values of the forget
         *  gates will create a vanishing gradients problem."
         *  http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
         */

        if (initializeParams) {
            //As per standard LSTM
            val fanIn = nL;
            val fanOut = nLast + nL;
            val inputWShape = new long[] {nLast, 4 * nL};
            val recurrentWShape = new long[] {nL, 4 * nL + 3};

            params.put(INPUT_WEIGHT_KEY_FORWARDS, WeightInitUtil.initWeights(fanIn, fanOut, inputWShape,
                            layerConf.getWeightInit(), dist, iwF));
            params.put(RECURRENT_WEIGHT_KEY_FORWARDS, WeightInitUtil.initWeights(fanIn, fanOut, recurrentWShape,
                            layerConf.getWeightInit(), dist, rwF));
            params.put(BIAS_KEY_FORWARDS, bF);
            params.put(INPUT_WEIGHT_KEY_BACKWARDS, WeightInitUtil.initWeights(fanIn, fanOut, inputWShape,
                            layerConf.getWeightInit(), dist, iwR));
            params.put(RECURRENT_WEIGHT_KEY_BACKWARDS, WeightInitUtil.initWeights(fanIn, fanOut, recurrentWShape,
                            layerConf.getWeightInit(), dist, rwR));
            params.put(BIAS_KEY_BACKWARDS, bR);
        } else {
            params.put(INPUT_WEIGHT_KEY_FORWARDS, WeightInitUtil.reshapeWeights(new long[] {nLast, 4 * nL}, iwF));
            params.put(RECURRENT_WEIGHT_KEY_FORWARDS, WeightInitUtil.reshapeWeights(new long[] {nL, 4 * nL + 3}, rwF));
            params.put(BIAS_KEY_FORWARDS, bF);
            params.put(INPUT_WEIGHT_KEY_BACKWARDS, WeightInitUtil.reshapeWeights(new long[] {nLast, 4 * nL}, iwR));
            params.put(RECURRENT_WEIGHT_KEY_BACKWARDS, WeightInitUtil.reshapeWeights(new long[] {nL, 4 * nL + 3}, rwR));
            params.put(BIAS_KEY_BACKWARDS, bR);
        }

        return params;
    }


    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM layerConf =
                        (org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM) conf.getLayer();

        val nL = layerConf.getNOut(); //i.e., n neurons in this layer
        val nLast = layerConf.getNIn(); //i.e., n neurons in previous layer

        val nParamsInput = nLast * (4 * nL);
        val nParamsRecurrent = nL * (4 * nL + 3);
        val nBias = 4 * nL;

        val rwFOffset = nParamsInput;
        val bFOffset = rwFOffset + nParamsRecurrent;
        val iwROffset = bFOffset + nBias;
        val rwROffset = iwROffset + nParamsInput;
        val bROffset = rwROffset + nParamsRecurrent;

        INDArray iwFG = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, rwFOffset)).reshape('f', nLast,
                        4 * nL);
        INDArray rwFG = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(rwFOffset, bFOffset)).reshape('f',
                        nL, 4 * nL + 3);
        INDArray bFG = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(bFOffset, iwROffset));
        INDArray iwRG = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(iwROffset, rwROffset))
                        .reshape('f', nLast, 4 * nL);
        INDArray rwRG = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(rwROffset, bROffset)).reshape('f',
                        nL, 4 * nL + 3);
        INDArray bRG = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(bROffset, bROffset + nBias));

        Map<String, INDArray> out = new LinkedHashMap<>();
        out.put(INPUT_WEIGHT_KEY_FORWARDS, iwFG);
        out.put(RECURRENT_WEIGHT_KEY_FORWARDS, rwFG);
        out.put(BIAS_KEY_FORWARDS, bFG);
        out.put(INPUT_WEIGHT_KEY_BACKWARDS, iwRG);
        out.put(RECURRENT_WEIGHT_KEY_BACKWARDS, rwRG);
        out.put(BIAS_KEY_BACKWARDS, bRG);

        return out;
    }
}
