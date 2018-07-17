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
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Parameter initializer for the Variational Autoencoder model.
 *
 * See: Kingma & Welling, 2013: Auto-Encoding Variational Bayes - https://arxiv.org/abs/1312.6114
 *
 * @author Alex Black
 */
public class VariationalAutoencoderParamInitializer extends DefaultParamInitializer {

    private static final VariationalAutoencoderParamInitializer INSTANCE = new VariationalAutoencoderParamInitializer();

    public static VariationalAutoencoderParamInitializer getInstance() {
        return INSTANCE;
    }

    public static final String WEIGHT_KEY_SUFFIX = "W";
    public static final String BIAS_KEY_SUFFIX = "b";
    public static final String PZX_PREFIX = "pZX";
    public static final String PZX_MEAN_PREFIX = PZX_PREFIX + "Mean";
    public static final String PZX_LOGSTD2_PREFIX = PZX_PREFIX + "LogStd2";

    public static final String ENCODER_PREFIX = "e";
    public static final String DECODER_PREFIX = "d";

    /** Key for weight parameters connecting the last encoder layer and the mean values for p(z|data) */
    public static final String PZX_MEAN_W = "pZXMean" + WEIGHT_KEY_SUFFIX;
    /** Key for bias parameters for the mean values for p(z|data) */
    public static final String PZX_MEAN_B = "pZXMean" + BIAS_KEY_SUFFIX;
    /** Key for weight parameters connecting the last encoder layer and the log(sigma^2) values for p(z|data) */
    public static final String PZX_LOGSTD2_W = PZX_LOGSTD2_PREFIX + WEIGHT_KEY_SUFFIX;
    /** Key for bias parameters for log(sigma^2) in p(z|data) */
    public static final String PZX_LOGSTD2_B = PZX_LOGSTD2_PREFIX + BIAS_KEY_SUFFIX;

    public static final String PXZ_PREFIX = "pXZ";
    /** Key for weight parameters connecting the last decoder layer and p(data|z) (according to whatever
     *  {@link org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution} is set for the VAE) */
    public static final String PXZ_W = PXZ_PREFIX + WEIGHT_KEY_SUFFIX;
    /** Key for bias parameters connecting the last decoder layer and p(data|z) (according to whatever
     *  {@link org.deeplearning4j.nn.conf.layers.variational.ReconstructionDistribution} is set for the VAE) */
    public static final String PXZ_B = PXZ_PREFIX + BIAS_KEY_SUFFIX;



    @Override
    public long numParams(NeuralNetConfiguration conf) {
        VariationalAutoencoder layer = (VariationalAutoencoder) conf.getLayer();

        val nIn = layer.getNIn();
        val nOut = layer.getNOut();
        int[] encoderLayerSizes = layer.getEncoderLayerSizes();
        int[] decoderLayerSizes = layer.getDecoderLayerSizes();

        int paramCount = 0;
        for (int i = 0; i < encoderLayerSizes.length; i++) {
            long encoderLayerIn;
            if (i == 0) {
                encoderLayerIn = nIn;
            } else {
                encoderLayerIn = encoderLayerSizes[i - 1];
            }
            paramCount += (encoderLayerIn + 1) * encoderLayerSizes[i]; //weights + bias
        }

        //Between the last encoder layer and the parameters for p(z|x):
        int lastEncLayerSize = encoderLayerSizes[encoderLayerSizes.length - 1];
        paramCount += (lastEncLayerSize + 1) * 2 * nOut; //Mean and variance parameters used in unsupervised training

        //Decoder:
        for (int i = 0; i < decoderLayerSizes.length; i++) {
            long decoderLayerNIn;
            if (i == 0) {
                decoderLayerNIn = nOut;
            } else {
                decoderLayerNIn = decoderLayerSizes[i - 1];
            }
            paramCount += (decoderLayerNIn + 1) * decoderLayerSizes[i];
        }

        //Between last decoder layer and parameters for p(x|z):
        // FIXME: int cast
        val nDistributionParams = layer.getOutputDistribution().distributionInputSize((int) nIn);
        val lastDecLayerSize = decoderLayerSizes[decoderLayerSizes.length - 1];
        paramCount += (lastDecLayerSize + 1) * nDistributionParams;

        return paramCount;
    }

    @Override
    public List<String> paramKeys(Layer l) {
        VariationalAutoencoder layer = (VariationalAutoencoder) l;
        int[] encoderLayerSizes = layer.getEncoderLayerSizes();
        int[] decoderLayerSizes = layer.getDecoderLayerSizes();

        List<String> p = new ArrayList<>();

        int soFar = 0;
        for (int i = 0; i < encoderLayerSizes.length; i++) {
            String sW = "e" + i + WEIGHT_KEY_SUFFIX;
            String sB = "e" + i + BIAS_KEY_SUFFIX;
            p.add(sW);
            p.add(sB);
        }

        //Last encoder layer -> p(z|x)
        p.add(PZX_MEAN_W);
        p.add(PZX_MEAN_B);

        //Pretrain params
        p.add(PZX_LOGSTD2_W);
        p.add(PZX_LOGSTD2_B);

        for (int i = 0; i < decoderLayerSizes.length; i++) {
            String sW = "d" + i + WEIGHT_KEY_SUFFIX;
            String sB = "d" + i + BIAS_KEY_SUFFIX;
            p.add(sW);
            p.add(sB);
        }

        //Finally, p(x|z):
        p.add(PXZ_W);
        p.add(PXZ_B);

        return p;
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        List<String> out = new ArrayList<>();
        for(String s : paramKeys(layer)){
            if(isWeightParam(layer, s)){
                out.add(s);
            }
        }
        return out;
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        List<String> out = new ArrayList<>();
        for(String s : paramKeys(layer)){
            if(isBiasParam(layer, s)){
                out.add(s);
            }
        }
        return out;
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return key.endsWith(WEIGHT_KEY_SUFFIX);
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return key.endsWith(BIAS_KEY_SUFFIX);
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        if (paramsView.length() != numParams(conf)) {
            throw new IllegalArgumentException("Incorrect paramsView length: Expected length " + numParams(conf)
                            + ", got length " + paramsView.length());
        }

        Map<String, INDArray> ret = new LinkedHashMap<>();
        VariationalAutoencoder layer = (VariationalAutoencoder) conf.getLayer();

        val nIn = layer.getNIn();
        val nOut = layer.getNOut();
        int[] encoderLayerSizes = layer.getEncoderLayerSizes();
        int[] decoderLayerSizes = layer.getDecoderLayerSizes();

        WeightInit weightInit = layer.getWeightInit();
        Distribution dist = Distributions.createDistribution(layer.getDist());

        int soFar = 0;
        for (int i = 0; i < encoderLayerSizes.length; i++) {
            long encoderLayerNIn;
            if (i == 0) {
                encoderLayerNIn = nIn;
            } else {
                encoderLayerNIn = encoderLayerSizes[i - 1];
            }
            val weightParamCount = encoderLayerNIn * encoderLayerSizes[i];
            INDArray weightView = paramsView.get(NDArrayIndex.point(0),
                            NDArrayIndex.interval(soFar, soFar + weightParamCount));
            soFar += weightParamCount;
            INDArray biasView = paramsView.get(NDArrayIndex.point(0),
                            NDArrayIndex.interval(soFar, soFar + encoderLayerSizes[i]));
            soFar += encoderLayerSizes[i];

            INDArray layerWeights = createWeightMatrix(encoderLayerNIn, encoderLayerSizes[i], weightInit, dist,
                            weightView, initializeParams);
            INDArray layerBiases = createBias(encoderLayerSizes[i], 0.0, biasView, initializeParams); //TODO don't hardcode 0

            String sW = "e" + i + WEIGHT_KEY_SUFFIX;
            String sB = "e" + i + BIAS_KEY_SUFFIX;
            ret.put(sW, layerWeights);
            ret.put(sB, layerBiases);

            conf.addVariable(sW);
            conf.addVariable(sB);
        }

        //Last encoder layer -> p(z|x)
        val nWeightsPzx = encoderLayerSizes[encoderLayerSizes.length - 1] * nOut;
        INDArray pzxWeightsMean =
                        paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + nWeightsPzx));
        soFar += nWeightsPzx;
        INDArray pzxBiasMean = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + nOut));
        soFar += nOut;

        INDArray pzxWeightsMeanReshaped = createWeightMatrix(encoderLayerSizes[encoderLayerSizes.length - 1], nOut,
                        weightInit, dist, pzxWeightsMean, initializeParams);
        INDArray pzxBiasMeanReshaped = createBias(nOut, 0.0, pzxBiasMean, initializeParams); //TODO don't hardcode 0

        ret.put(PZX_MEAN_W, pzxWeightsMeanReshaped);
        ret.put(PZX_MEAN_B, pzxBiasMeanReshaped);
        conf.addVariable(PZX_MEAN_W);
        conf.addVariable(PZX_MEAN_B);


        //Pretrain params
        INDArray pzxWeightsLogStdev2 =
                        paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + nWeightsPzx));
        soFar += nWeightsPzx;
        INDArray pzxBiasLogStdev2 = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + nOut));
        soFar += nOut;

        INDArray pzxWeightsLogStdev2Reshaped = createWeightMatrix(encoderLayerSizes[encoderLayerSizes.length - 1], nOut,
                        weightInit, dist, pzxWeightsLogStdev2, initializeParams);
        INDArray pzxBiasLogStdev2Reshaped = createBias(nOut, 0.0, pzxBiasLogStdev2, initializeParams); //TODO don't hardcode 0

        ret.put(PZX_LOGSTD2_W, pzxWeightsLogStdev2Reshaped);
        ret.put(PZX_LOGSTD2_B, pzxBiasLogStdev2Reshaped);
        conf.addVariable(PZX_LOGSTD2_W);
        conf.addVariable(PZX_LOGSTD2_B);

        for (int i = 0; i < decoderLayerSizes.length; i++) {
            long decoderLayerNIn;
            if (i == 0) {
                decoderLayerNIn = nOut;
            } else {
                decoderLayerNIn = decoderLayerSizes[i - 1];
            }
            val weightParamCount = decoderLayerNIn * decoderLayerSizes[i];
            INDArray weightView = paramsView.get(NDArrayIndex.point(0),
                            NDArrayIndex.interval(soFar, soFar + weightParamCount));
            soFar += weightParamCount;
            INDArray biasView = paramsView.get(NDArrayIndex.point(0),
                            NDArrayIndex.interval(soFar, soFar + decoderLayerSizes[i]));
            soFar += decoderLayerSizes[i];

            INDArray layerWeights = createWeightMatrix(decoderLayerNIn, decoderLayerSizes[i], weightInit, dist,
                            weightView, initializeParams);
            INDArray layerBiases = createBias(decoderLayerSizes[i], 0.0, biasView, initializeParams); //TODO don't hardcode 0

            String sW = "d" + i + WEIGHT_KEY_SUFFIX;
            String sB = "d" + i + BIAS_KEY_SUFFIX;
            ret.put(sW, layerWeights);
            ret.put(sB, layerBiases);
            conf.addVariable(sW);
            conf.addVariable(sB);
        }

        //Finally, p(x|z):
        // FIXME: int cast
        int nDistributionParams = layer.getOutputDistribution().distributionInputSize((int) nIn);
        int pxzWeightCount = decoderLayerSizes[decoderLayerSizes.length - 1] * nDistributionParams;
        INDArray pxzWeightView =
                        paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + pxzWeightCount));
        soFar += pxzWeightCount;
        INDArray pxzBiasView = paramsView.get(NDArrayIndex.point(0),
                        NDArrayIndex.interval(soFar, soFar + nDistributionParams));

        INDArray pxzWeightsReshaped = createWeightMatrix(decoderLayerSizes[decoderLayerSizes.length - 1],
                        nDistributionParams, weightInit, dist, pxzWeightView, initializeParams);
        INDArray pxzBiasReshaped = createBias(nDistributionParams, 0.0, pxzBiasView, initializeParams); //TODO don't hardcode 0

        ret.put(PXZ_W, pxzWeightsReshaped);
        ret.put(PXZ_B, pxzBiasReshaped);
        conf.addVariable(PXZ_W);
        conf.addVariable(PXZ_B);

        return ret;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        Map<String, INDArray> ret = new LinkedHashMap<>();
        VariationalAutoencoder layer = (VariationalAutoencoder) conf.getLayer();

        val nIn = layer.getNIn();
        val nOut = layer.getNOut();
        int[] encoderLayerSizes = layer.getEncoderLayerSizes();
        int[] decoderLayerSizes = layer.getDecoderLayerSizes();

        int soFar = 0;
        for (int i = 0; i < encoderLayerSizes.length; i++) {
            long encoderLayerNIn;
            if (i == 0) {
                encoderLayerNIn = nIn;
            } else {
                encoderLayerNIn = encoderLayerSizes[i - 1];
            }
            val weightParamCount = encoderLayerNIn * encoderLayerSizes[i];
            INDArray weightGradView = gradientView.get(NDArrayIndex.point(0),
                            NDArrayIndex.interval(soFar, soFar + weightParamCount));
            soFar += weightParamCount;
            INDArray biasGradView = gradientView.get(NDArrayIndex.point(0),
                            NDArrayIndex.interval(soFar, soFar + encoderLayerSizes[i]));
            soFar += encoderLayerSizes[i];

            INDArray layerWeights = weightGradView.reshape('f', encoderLayerNIn, encoderLayerSizes[i]);
            INDArray layerBiases = biasGradView; //Aready correct shape (row vector)

            ret.put("e" + i + WEIGHT_KEY_SUFFIX, layerWeights);
            ret.put("e" + i + BIAS_KEY_SUFFIX, layerBiases);
        }

        //Last encoder layer -> p(z|x)
        val nWeightsPzx = encoderLayerSizes[encoderLayerSizes.length - 1] * nOut;
        INDArray pzxWeightsMean =
                        gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + nWeightsPzx));
        soFar += nWeightsPzx;
        INDArray pzxBiasMean = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + nOut));
        soFar += nOut;

        INDArray pzxWeightGradMeanReshaped =
                        pzxWeightsMean.reshape('f', encoderLayerSizes[encoderLayerSizes.length - 1], nOut);

        ret.put(PZX_MEAN_W, pzxWeightGradMeanReshaped);
        ret.put(PZX_MEAN_B, pzxBiasMean);

        ////////////////////////////////////////////////////////

        INDArray pzxWeightsLogStdev2 =
                        gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + nWeightsPzx));
        soFar += nWeightsPzx;
        INDArray pzxBiasLogStdev2 = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + nOut));
        soFar += nOut;

        INDArray pzxWeightsLogStdev2Reshaped = createWeightMatrix(encoderLayerSizes[encoderLayerSizes.length - 1], nOut,
                        null, null, pzxWeightsLogStdev2, false); //TODO

        ret.put(PZX_LOGSTD2_W, pzxWeightsLogStdev2Reshaped);
        ret.put(PZX_LOGSTD2_B, pzxBiasLogStdev2);

        for (int i = 0; i < decoderLayerSizes.length; i++) {
            long decoderLayerNIn;
            if (i == 0) {
                decoderLayerNIn = nOut;
            } else {
                decoderLayerNIn = decoderLayerSizes[i - 1];
            }
            long weightParamCount = decoderLayerNIn * decoderLayerSizes[i];
            INDArray weightView = gradientView.get(NDArrayIndex.point(0),
                            NDArrayIndex.interval(soFar, soFar + weightParamCount));
            soFar += weightParamCount;
            INDArray biasView = gradientView.get(NDArrayIndex.point(0),
                            NDArrayIndex.interval(soFar, soFar + decoderLayerSizes[i]));
            soFar += decoderLayerSizes[i];

            INDArray layerWeights =
                            createWeightMatrix(decoderLayerNIn, decoderLayerSizes[i], null, null, weightView, false);
            INDArray layerBiases = createBias(decoderLayerSizes[i], 0.0, biasView, false); //TODO don't hardcode 0

            String sW = "d" + i + WEIGHT_KEY_SUFFIX;
            String sB = "d" + i + BIAS_KEY_SUFFIX;
            ret.put(sW, layerWeights);
            ret.put(sB, layerBiases);
        }

        //Finally, p(x|z):
        // FIXME: int cast
        int nDistributionParams = layer.getOutputDistribution().distributionInputSize((int) nIn);
        int pxzWeightCount = decoderLayerSizes[decoderLayerSizes.length - 1] * nDistributionParams;
        INDArray pxzWeightView =
                        gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + pxzWeightCount));
        soFar += pxzWeightCount;
        INDArray pxzBiasView = gradientView.get(NDArrayIndex.point(0),
                        NDArrayIndex.interval(soFar, soFar + nDistributionParams));

        INDArray pxzWeightsReshaped = createWeightMatrix(decoderLayerSizes[decoderLayerSizes.length - 1],
                        nDistributionParams, null, null, pxzWeightView, false);
        INDArray pxzBiasReshaped = createBias(nDistributionParams, 0.0, pxzBiasView, false);

        ret.put(PXZ_W, pxzWeightsReshaped);
        ret.put(PXZ_B, pxzBiasReshaped);

        return ret;
    }
}
