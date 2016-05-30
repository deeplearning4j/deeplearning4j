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

package org.deeplearning4j.nn.params;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.LinkedHashMap;
import java.util.Map;

/**LSTM Parameter initializer, for LSTM based on
 * Graves: Supervised Sequence Labelling with Recurrent Neural Networks
 * http://www.cs.toronto.edu/~graves/phd.pdf
 */
public class GravesBidirectionalLSTMParamInitializer implements ParamInitializer {
	/** Weights for previous time step -> current time step connections */
    public final static String RECURRENT_WEIGHT_KEY_FORWARDS = "RWF";
    public final static String BIAS_KEY_FORWARDS = DefaultParamInitializer.BIAS_KEY + "F";
    public final static String INPUT_WEIGHT_KEY_FORWARDS = DefaultParamInitializer.WEIGHT_KEY + "F";

    public final static String RECURRENT_WEIGHT_KEY_BACKWARDS = "RWB";
    public final static String BIAS_KEY_BACKWARDS = DefaultParamInitializer.BIAS_KEY + "B";
    public final static String INPUT_WEIGHT_KEY_BACKWARDS = DefaultParamInitializer.WEIGHT_KEY + "B";

    @Override
    public int numParams(NeuralNetConfiguration conf, boolean backprop) {
        org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM layerConf =
                (org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM) conf.getLayer();

        int nL = layerConf.getNOut();	//i.e., n neurons in this layer
        int nLast = layerConf.getNIn();	//i.e., n neurons in previous layer

        int nParamsForward = nLast * (4*nL)   //"input" weights
                + nL * (4 * nL + 3) //recurrent weights
                + 4*nL;             //bias

        return 2*nParamsForward;
    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, INDArray paramsView) {
        org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM layerConf =
                (org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM) conf.getLayer();
        double forgetGateInit = layerConf.getForgetGateBiasInit();

        Distribution dist = Distributions.createDistribution(layerConf.getDist());

        int nL = layerConf.getNOut();	//i.e., n neurons in this layer
        int nLast = layerConf.getNIn();	//i.e., n neurons in previous layer
        
        conf.addVariable(INPUT_WEIGHT_KEY_FORWARDS);
        conf.addVariable(RECURRENT_WEIGHT_KEY_FORWARDS);
        conf.addVariable(BIAS_KEY_FORWARDS);
        conf.addVariable(INPUT_WEIGHT_KEY_BACKWARDS);
        conf.addVariable(RECURRENT_WEIGHT_KEY_BACKWARDS);
        conf.addVariable(BIAS_KEY_BACKWARDS);

        int nParamsInput = nLast * (4*nL);
        int nParamsRecurrent = nL * (4 * nL + 3);
        int nBias = 4*nL;

        int rwFOffset = nParamsInput;
        int bFOffset = rwFOffset + nParamsRecurrent;
        int iwROffset = bFOffset + nBias;
        int rwROffset = iwROffset + nParamsInput;
        int bROffset = rwROffset + nParamsRecurrent;

        INDArray iwF = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, rwFOffset));
        INDArray rwF = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(rwFOffset, bFOffset));
        INDArray bF = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(bFOffset, iwROffset));
        INDArray iwR = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(iwROffset, rwROffset));
        INDArray rwR = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(rwROffset, bROffset));
        INDArray bR = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(bROffset, bROffset + nBias));

        bF.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.interval(nL, 2*nL)}, Nd4j.ones(1,nL).muli(forgetGateInit)); //Order: input, forget, output, input modulation, i.e., IFOG
        bR.put(new INDArrayIndex[]{NDArrayIndex.point(0),NDArrayIndex.interval(nL, 2*nL)}, Nd4j.ones(1,nL).muli(forgetGateInit));
        /*The above line initializes the forget gate biases to specified value.
         * See Sutskever PhD thesis, pg19:
         * "it is important for [the forget gate activations] to be approximately 1 at the early stages of learning,
         *  which is accomplished by initializing [the forget gate biases] to a large value (such as 5). If it is
         *  not done, it will be harder to learn long range dependencies because the smaller values of the forget
         *  gates will create a vanishing gradients problem."
         *  http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
         */
        
        params.put(INPUT_WEIGHT_KEY_FORWARDS,WeightInitUtil.initWeights(nLast, 4 * nL, layerConf.getWeightInit(), dist, iwF));
        params.put(RECURRENT_WEIGHT_KEY_FORWARDS,WeightInitUtil.initWeights(nL, 4 * nL + 3, layerConf.getWeightInit(), dist, rwF));
        params.put(BIAS_KEY_FORWARDS, bF);
        params.put(INPUT_WEIGHT_KEY_BACKWARDS,WeightInitUtil.initWeights(nLast, 4 * nL, layerConf.getWeightInit(), dist, iwR));
        params.put(RECURRENT_WEIGHT_KEY_BACKWARDS,WeightInitUtil.initWeights(nL, 4 * nL + 3, layerConf.getWeightInit(), dist, rwR));
        params.put(BIAS_KEY_BACKWARDS,bR);
    }


    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM layerConf = (org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM) conf.getLayer();

        int nL = layerConf.getNOut();	//i.e., n neurons in this layer
        int nLast = layerConf.getNIn();	//i.e., n neurons in previous layer

        int nParamsInput = nLast * (4*nL);
        int nParamsRecurrent = nL * (4 * nL + 3);
        int nBias = 4*nL;

        int rwFOffset = nParamsInput;
        int bFOffset = rwFOffset + nParamsRecurrent;
        int iwROffset = bFOffset + nBias;
        int rwROffset = iwROffset + nParamsInput;
        int bROffset = rwROffset + nParamsRecurrent;

        INDArray iwFG = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, rwFOffset)).reshape('f',nLast, 4*nL);
        INDArray rwFG = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(rwFOffset, bFOffset)).reshape('f',nL, 4*nL+3);
        INDArray bFG = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(bFOffset, iwROffset));
        INDArray iwRG = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(iwROffset, rwROffset)).reshape('f',nLast, 4*nL);
        INDArray rwRG = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(rwROffset, bROffset)).reshape('f',nL, 4*nL+3);
        INDArray bRG = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(bROffset, bROffset + nBias));

        Map<String,INDArray> out = new LinkedHashMap<>();
        out.put(INPUT_WEIGHT_KEY_FORWARDS, iwFG);
        out.put(RECURRENT_WEIGHT_KEY_FORWARDS, rwFG);
        out.put(BIAS_KEY_FORWARDS, bFG);
        out.put(INPUT_WEIGHT_KEY_BACKWARDS, iwRG);
        out.put(RECURRENT_WEIGHT_KEY_BACKWARDS, rwRG);
        out.put(BIAS_KEY_BACKWARDS, bRG);

        return out;
    }
}
