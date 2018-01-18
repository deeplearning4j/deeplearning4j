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

package org.deeplearning4j.nn.layers.convolution.subsampling;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.deeplearning4j.util.OneTimeLogger;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;
import java.util.Properties;


/**
 * Subsampling layer.
 *
 * Used for downsampling a convolution
 *
 * @author Adam Gibson
 */
@Slf4j
public class SubsamplingLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.SubsamplingLayer> {

    protected SubsamplingHelper helper = null;
    protected ConvolutionMode convolutionMode;

    public SubsamplingLayer(NeuralNetConfiguration conf) {
        super(conf);
        initializeHelper();
        this.convolutionMode =
                        ((org.deeplearning4j.nn.conf.layers.SubsamplingLayer) conf.getLayer()).getConvolutionMode();
    }

    public SubsamplingLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
        initializeHelper();
    }

    void initializeHelper() {
        try {
            helper = Class.forName("org.deeplearning4j.nn.layers.convolution.subsampling.CudnnSubsamplingHelper")
                            .asSubclass(SubsamplingHelper.class).newInstance();
            log.debug("CudnnSubsamplingHelper successfully initialized");
            if (!helper.checkSupported()) {
                helper = null;
            }
        } catch (Throwable t) {
            if (!(t instanceof ClassNotFoundException)) {
                log.warn("Could not initialize CudnnSubsamplingHelper", t);
            } else {
                Properties p = Nd4j.getExecutioner().getEnvironmentInformation();
                if (p.getProperty("backend").equals("CUDA")) {
                    OneTimeLogger.info(log, "cuDNN not found: "
                                    + "use cuDNN for better GPU performance by including the deeplearning4j-cuda module. "
                                    + "For more information, please refer to: https://deeplearning4j.org/cudnn", t);
                }
            }
        }
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public Type type() {
        return Type.SUBSAMPLING;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        int miniBatch = input.size(0);
        int inDepth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] dilation = layerConf().getDilation();

        int[] pad;
        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode, dilation); //Also performs validation
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {inH, inW}, kernel, strides, dilation);
        } else {
            pad = layerConf().getPadding();
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode, dilation); //Also performs validation
        }

        if (helper != null) {
            Pair<Gradient, INDArray> ret = helper.backpropGradient(input, epsilon, kernel, strides, pad,
                            layerConf().getPoolingType(), convolutionMode, dilation);
            if (ret != null) {
                return ret;
            }
        }

        //subsampling doesn't have weights and thus gradients are not calculated for this layer
        Gradient retGradient = new DefaultGradient();


        PoolingType pt = layerConf().getPoolingType();
        String fn;
        int nIArgs;
        int nTArgs;
        switch (pt){
            case MAX:
                fn = "maxpool2d_bp";    //https://github.com/deeplearning4j/libnd4j/blob/master/include/ops/declarable/generic/convo/pooling/maxpool2d.cpp
                nIArgs = 11;
                nTArgs = 0;
                break;
            case AVG:
                fn = "avgpool2d_bp";    //https://github.com/deeplearning4j/libnd4j/blob/master/include/ops/declarable/generic/convo/pooling/avgpool2d.cpp
                nIArgs = 9;
                nTArgs = 0;
                break;
            case PNORM:
                fn = "pnormpool2d_bp";  //https://github.com/deeplearning4j/libnd4j/blob/master/include/ops/declarable/generic/convo/pooling/pnormpool2d.cpp
                nIArgs = 10;
                nTArgs = 1;
                break;
            case SUM:
            default:
                throw new RuntimeException("Not supported: " + layerConf().getPoolingType());
        }

        int[] a = new int[nIArgs];
        a[0] = kernel[0];
        a[1] = kernel[1];
        a[2] = strides[0];
        a[3] = strides[1];
        a[4] = pad[0];
        a[5] = pad[1];
        a[6] = dilation[0];
        a[7] = dilation[1];
        a[8] = layerConf().getConvolutionMode() == ConvolutionMode.Same ? 1 : 0;

        Double[] d = new Double[nTArgs];

        if(pt == PoolingType.MAX){
            //a[9]: Not used with max pooling
            a[10] = 0;  //For NCHW
        } else if(pt == PoolingType.PNORM){
            a[9] = layerConf().getPnorm();
            d[0] = layerConf().getEps();
        }

        INDArray epsNext = Nd4j.create(input.shape(), 'c');

        DynamicCustomOp op = DynamicCustomOp.builder(fn)
                .addInputs(input, epsilon)
                .addOutputs(epsNext)
                .addIntegerArguments(a)
                .addFloatingPointArguments(d)
                .build();

        Nd4j.getExecutioner().exec(op);

        return new Pair<>(retGradient, epsNext);
    }


    @Override
    public INDArray activate(boolean training) {
        if (training && !dropoutApplied && layerConf().getIDropout() != null) {
            applyDropOutIfNecessary(true);
        }

        //Input validation: expect rank 4 matrix
        if (input.rank() != 4) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                            + " array as input to SubsamplingLayer with shape " + Arrays.toString(input.shape())
                            + ". Expected rank 4 array with shape [minibatchSize, depth, inputHeight, inputWidth]. "
                            + layerId());
        }

        int miniBatch = input.size(0);
        int inDepth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] dilation = layerConf().getDilation();
        int[] pad;
        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode, dilation); //Also performs validation
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {inH, inW}, kernel, strides, dilation);
        } else {
            pad = layerConf().getPadding();
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode, dilation); //Also performs validation
        }
        int outH = outSize[0];
        int outW = outSize[1];

        if (helper != null) {
            INDArray ret = helper.activate(input, training, kernel, strides, pad, layerConf().getPoolingType(),
                            convolutionMode, dilation);
            if (ret != null) {
                return ret;
            }
        }

        //Similar to convolution layer forward pass: do im2col, but permute so that pooling can be done with efficient strides...
        //Current im2col implementation expects input with shape [miniBatch,depth,kH,kW,outH,outW]

        INDArray output = Nd4j.create(miniBatch, inDepth, outH, outW);


        switch (layerConf().getPoolingType()) {
            case AVG:
                //                reduced = col2d.mean(1);
                //                time2 = System.nanoTime();

                Convolution.pooling2D(input, kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                                convolutionMode == ConvolutionMode.Same, Pooling2D.Pooling2DType.AVG, Pooling2D.Divisor.INCLUDE_PADDING,
                                0.0, outH, outW, output);

                break;
            case MAX:
                Convolution.pooling2D(input, kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                                convolutionMode == ConvolutionMode.Same, Pooling2D.Pooling2DType.MAX, Pooling2D.Divisor.INCLUDE_PADDING,
                                0.0, outH, outW, output);

                break;
            case PNORM:
                // pnorm pooling is used for signal loss recovery it is mixed with avg pooling,
                // applying the exponent to the input and recovering the signal by multiplying the kernel of
                // the pooling layer and then applying the same inverse exponent
                int pnorm = layerConf().getPnorm();
                /*
                
                Transforms.abs(col2d, false);
                Transforms.pow(col2d, pnorm, false);
                reduced = col2d.sum(1);
                Transforms.pow(reduced, (1.0 / pnorm), false);
                time2 = System.nanoTime();
                */

                Convolution.pooling2D(input, kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                                convolutionMode == ConvolutionMode.Same, Pooling2D.Pooling2DType.PNORM, Pooling2D.Divisor.INCLUDE_PADDING,
                                (double) pnorm, outH, outW, output);

                break;
            default:
                throw new IllegalStateException("Unknown/not supported pooling type: " + layerConf().getPoolingType()
                                + " " + layerId());
        }

        return output;
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException(layerId());
    }

    @Override
    public Layer clone() {
        return new SubsamplingLayer(conf.clone());
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {
        //no op
    }

    @Override
    public void iterate(INDArray input) {
        throw new UnsupportedOperationException(layerId());
    }

    @Override
    public Gradient gradient() {
        throw new UnsupportedOperationException("Not supported - no parameters");
    }

    @Override
    public void fit() {

    }

    @Override
    public int numParams() {
        return 0;
    }

    @Override
    public void fit(INDArray input) {}

    @Override
    public void computeGradientAndScore() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public double score() {
        return 0;
    }

    @Override
    public void accumulateScore(double accum) {
        throw new UnsupportedOperationException(layerId());
    }


    @Override
    public void update(INDArray gradient, String paramType) {

    }

    @Override
    public INDArray params() {
        return null;
    }

    @Override
    public INDArray getParam(String param) {
        return params();
    }

    @Override
    public void setParams(INDArray params) {

    }

    @Override
    public INDArray preOutput(boolean training) {
        return activate(training);
    }


}
