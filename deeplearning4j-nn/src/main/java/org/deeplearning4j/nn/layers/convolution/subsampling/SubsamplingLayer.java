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
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.api.ops.impl.transforms.convolution.Pooling2D;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;


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
        int outH = outSize[0];
        int outW = outSize[1];


        if (helper != null) {
            Pair<Gradient, INDArray> ret = helper.backpropGradient(input, epsilon, kernel, strides, pad,
                            layerConf().getPoolingType(), convolutionMode, dilation);
            if (ret != null) {
                return ret;
            }
        }

        //subsampling doesn't have weights and thus gradients are not calculated for this layer
        //only scale and reshape epsilon
        int inputHeight = input().size(-2);
        int inputWidth = input().size(-1);
        Gradient retGradient = new DefaultGradient();

        //Epsilons in shape: [miniBatch, depth, outH, outW]
        //Epsilons out shape: [miniBatch, depth, inH, inW]

        //Two possibilities here for the epsilons:
        //(a) Epsilons come from a dense/output layer above, with c order and strides [depth*H*W, H*W, W, 1]
        //(b) Epsilons come from CNN layer above, with c order and strides [H*W, depth*H*W, W, 1] (i.e., due to permute)

        //We want to reshape epsilons to 1d here, but to do this without a copy: we end up with different orders of
        // element in the buffer, for the "dense above" and "cnn above" cases.
        //Fortunately, we can just permute things when we do the im2col reshaping; then, the order of the rows in
        // col2d will match the order of the 1d epsilons...
        //With the 1d epsilons order matching the rows order for the 2d im2col: we can just do a muliColumnVector op,
        // instead of a slower broadcast muli op

        boolean cOrderStrides = false;
        if (epsilon.ordering() != 'c') {
            epsilon = epsilon.dup('c');
            cOrderStrides = true;
        }
        if (!cOrderStrides && Shape.strideDescendingCAscendingF(epsilon)) {
            cOrderStrides = true;
        } else if (!Arrays.equals(new int[] {outH * outW, inDepth * outH * outW, outW, 1}, epsilon.stride())) {
            //Unexpected/unusual strides, not either (a) or (b) cases above
            epsilon = epsilon.dup('c');
            cOrderStrides = true;
        }

        INDArray col6d;
        INDArray col6dPermuted;
        INDArray epsilon1d;
        if (cOrderStrides) {
            //"Dense/Output layer above strides... i.e., standard c-order strides
            col6d = Nd4j.create(new int[] {miniBatch, inDepth, outH, outW, kernel[0], kernel[1]}, 'c');
            col6dPermuted = col6d.permute(0, 1, 4, 5, 2, 3);
            epsilon1d = epsilon.reshape('c', ArrayUtil.prod(epsilon.length()), 1); //zero copy reshape
        } else {
            //"CNN layer above" strides...
            col6d = Nd4j.create(new int[] {inDepth, miniBatch, outH, outW, kernel[0], kernel[1]}, 'c');
            col6dPermuted = col6d.permute(1, 0, 4, 5, 2, 3);

            INDArray epsilonTemp = epsilon.permute(1, 0, 2, 3);
            epsilon1d = epsilonTemp.reshape('c', new int[] {ArrayUtil.prod(epsilon.length()), 1}); //Should be a zero-copy reshape always
        }

        INDArray col2d = col6d.reshape('c', miniBatch * inDepth * outH * outW, kernel[0] * kernel[1]);


        switch (layerConf().getPoolingType()) {
            case MAX:
                //Execute im2col, then reshape to 2d. Note rows are in a different order for cOrderStrides true vs false cases
                Convolution.im2col(input, kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                                convolutionMode == ConvolutionMode.Same, col6dPermuted);
                INDArray isMax = Nd4j.getExecutioner().execAndReturn(new IsMax(col2d, 1));
                isMax.muliColumnVector(epsilon1d);
                break;
            case AVG:
                //TODO: We could further optimize this by creating an uninitialized array, and doing a 'putiColumnVector' operation
                // instead of a zero initialization + an addiColumnVector op
                col2d.addiColumnVector(epsilon1d);
                break;
            case PNORM:
                int pnorm = layerConf().getPnorm();

                //First: do forward pass to get pNorm array
                Convolution.im2col(input, kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                                convolutionMode == ConvolutionMode.Same, col6dPermuted);
                INDArray pNorm = Transforms.abs(col2d, true); //dup as we need col2d again later
                Transforms.pow(pNorm, pnorm, false);
                pNorm = pNorm.sum(1);
                Transforms.pow(pNorm, (1.0 / pnorm), false);

                //dL/dIn = dL/dOut * dOut/dIn
                //dOut/dIn = in .* |in|^(p-2) /  ||in||_p^(p-1), where ||in||_p is the output p-norm
                INDArray numerator;
                if (pnorm == 2) {
                    numerator = col2d;
                } else {
                    INDArray absp2 = Transforms.pow(Transforms.abs(col2d, true), pnorm - 2, false);
                    numerator = col2d.muli(absp2);
                }

                INDArray denom = Transforms.pow(pNorm, pnorm - 1, false);
                double eps = layerConf().getEps();
                Transforms.max(denom, eps, false); // in case of 0
                numerator.muliColumnVector(denom.rdivi(epsilon1d));
                break;
            case NONE:
                return new Pair<>(retGradient, epsilon);
            default:
                throw new IllegalStateException("Unknown or unsupported pooling type: " + layerConf().getPoolingType()
                                + " " + layerId());
        }

        //Finally: we want the output strides for the epsilons to match the strides in the activations from the layer below
        //Assuming the layer below is a CNN layer (very likely) we want [H*W, depth*H*W, W, 1] instead of the standard
        // c-order [depth*H*W, H*W, W, 1] strides
        //To achieve this: [depth, miniBatch, H, W] in c order, then permute to [miniBatch, depth, H, W]
        //This gives us proper strides of 1 on the muli...
        INDArray tempEpsilon = Nd4j.create(new int[] {inDepth, miniBatch, inH, inW}, 'c');
        INDArray outEpsilon = tempEpsilon.permute(1, 0, 2, 3);
        Convolution.col2im(col6dPermuted, outEpsilon, strides[0], strides[1], pad[0], pad[1], inputHeight, inputWidth, dilation[0], dilation[1]);

        if (layerConf().getPoolingType() == PoolingType.AVG)
            outEpsilon.divi(ArrayUtil.prod(layerConf().getKernelSize()));
        return new Pair<>(retGradient, outEpsilon);
    }


    @Override
    public INDArray activate(boolean training) {
        if (training && !dropoutApplied && layerConf().getIDropout() != null) {
            //TODO: Epoch + iteration counters...
            if (Nd4j.getWorkspaceManager().checkIfWorkspaceExists(ComputationGraph.workspaceExternal)) {
                try (MemoryWorkspace ws = Nd4j.getWorkspaceManager()
                        .getWorkspaceForCurrentThread(ComputationGraph.workspaceExternal)
                        .notifyScopeBorrowed()) {
                    input = layerConf().getIDropout().applyDropout(input, -1, -1, false);
                }
            } else {
                input = layerConf().getIDropout().applyDropout(input, -1, -1, false);
            }
            dropoutApplied = true;
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

        INDArray output = Nd4j.createUninitialized(miniBatch * inDepth * outH * outW);


        switch (layerConf().getPoolingType()) {
            case AVG:
                //                reduced = col2d.mean(1);
                //                time2 = System.nanoTime();

                Convolution.pooling2D(input, kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                                convolutionMode == ConvolutionMode.Same, Pooling2D.Pooling2DType.AVG, 0.0, outH, outW,
                                output);

                break;
            case MAX:
                Convolution.pooling2D(input, kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                                convolutionMode == ConvolutionMode.Same, Pooling2D.Pooling2DType.MAX, 0.0, outH, outW,
                                output);

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
                                convolutionMode == ConvolutionMode.Same, Pooling2D.Pooling2DType.PNORM, (double) pnorm,
                                outH, outW, output);

                break;
            case NONE:
                return input;
            default:
                throw new IllegalStateException("Unknown/not supported pooling type: " + layerConf().getPoolingType()
                                + " " + layerId());
        }

        return output.reshape('c', miniBatch, inDepth, outH, outW);
    }

    @Override
    public Gradient error(INDArray input) {
        throw new UnsupportedOperationException(layerId());
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        throw new UnsupportedOperationException(layerId());
    }


    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException(layerId());
    }

    @Override
    public INDArray activationMean() {
        return null;
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
