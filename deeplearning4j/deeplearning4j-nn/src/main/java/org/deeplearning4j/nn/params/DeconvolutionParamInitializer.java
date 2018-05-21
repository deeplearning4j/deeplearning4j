package org.deeplearning4j.nn.params;

import lombok.val;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.LinkedHashMap;
import java.util.Map;

public class DeconvolutionParamInitializer extends ConvolutionParamInitializer {

    private static final DeconvolutionParamInitializer INSTANCE = new DeconvolutionParamInitializer();

    public static DeconvolutionParamInitializer getInstance() {
        return INSTANCE;
    }

    @Override
    protected INDArray createWeightMatrix(NeuralNetConfiguration conf, INDArray weightView, boolean initializeParams) {
        /*
         Create a 4d weight matrix of:
           (number of kernels, num input channels, kernel height, kernel width)
         Note c order is used specifically for the CNN weights, as opposed to f order elsewhere
         Inputs to the convolution layer are:
         (batch size, num input feature maps, image height, image width)
         */
        org.deeplearning4j.nn.conf.layers.Deconvolution2D layerConf =
                (org.deeplearning4j.nn.conf.layers.Deconvolution2D) conf.getLayer();
        if (initializeParams) {
            Distribution dist = Distributions.createDistribution(layerConf.getDist());
            int[] kernel = layerConf.getKernelSize();
            int[] stride = layerConf.getStride();

            val inputDepth = layerConf.getNIn();
            val outputDepth = layerConf.getNOut();

            double fanIn = inputDepth * kernel[0] * kernel[1];
            double fanOut = outputDepth * kernel[0] * kernel[1] / ((double) stride[0] * stride[1]);

            val weightsShape = new long[] {inputDepth, outputDepth, kernel[0], kernel[1]};

            INDArray weights = WeightInitUtil.initWeights(
                    fanIn, fanOut, weightsShape, layerConf.getWeightInit(), dist, 'c', weightView);

            return weights;
        } else {
            int[] kernel = layerConf.getKernelSize();

            INDArray weights =  WeightInitUtil.reshapeWeights(
                    new long[] {layerConf.getNIn(), layerConf.getNOut(), kernel[0],
                            kernel[1]}, weightView, 'c');

            return weights;
        }
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {

        org.deeplearning4j.nn.conf.layers.Deconvolution2D layerConf =
                (org.deeplearning4j.nn.conf.layers.Deconvolution2D) conf.getLayer();

        int[] kernel = layerConf.getKernelSize();
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();

        Map<String, INDArray> out = new LinkedHashMap<>();
        if(layerConf.hasBias()){
            INDArray biasGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut));
            INDArray weightGradientView =
                    gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nOut, numParams(conf)))
                            .reshape('c', nIn, nOut, kernel[0], kernel[1]);
            out.put(BIAS_KEY, biasGradientView);
            out.put(WEIGHT_KEY, weightGradientView);
        } else {
            INDArray weightGradientView = gradientView.reshape('c', nIn, nOut, kernel[0], kernel[1]);
            out.put(WEIGHT_KEY, weightGradientView);
        }
        return out;
    }
}
