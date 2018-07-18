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
import org.deeplearning4j.nn.layers.LayerHelper;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.layers.convolution.LegacyPooling2D;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.nd4j.util.OneTimeLogger;

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
    protected int helperCountFail = 0;
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
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        if("CUDA".equalsIgnoreCase(backend)) {
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
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);

        // FIXME: int cast
        int miniBatch = (int) input.size(0);
        int inDepth = (int) input.size(1);
        int inH = (int) input.size(2);
        int inW = (int) input.size(3);

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


        if (helper != null && (helperCountFail == 0 || !layerConf().isCudnnAllowFallback())) {
            Pair<Gradient, INDArray> ret = null;
            try{
                ret = helper.backpropGradient(input, epsilon, kernel, strides, pad,
                        layerConf().getPoolingType(), convolutionMode, dilation, workspaceMgr);
            } catch (Exception e){
                if(layerConf().isCudnnAllowFallback()){
                    helperCountFail++;
                    log.warn("CuDNN execution failed - falling back on built-in implementation",e);
                } else {
                    throw new RuntimeException(e);
                }
            }
            if (ret != null) {
                //Backprop dropout, if present
                INDArray gradPostDropout = ret.getRight();
                gradPostDropout = backpropDropOutIfPresent(gradPostDropout);
                ret.setSecond(gradPostDropout);
                return ret;
            }
        }

        //subsampling doesn't have weights and thus gradients are not calculated for this layer
        //only scale and reshape epsilon
        // FIXME: int cast
        int inputHeight = (int) input().size(-2);
        int inputWidth = (int) input().size(-1);
        Gradient retGradient = new DefaultGradient();

        //Epsilons in shape: [miniBatch, channels, outH, outW]
        //Epsilons out shape: [miniBatch, channels, inH, inW]

        //Two possibilities here for the epsilons:
        //(a) Epsilons come from a dense/output layer above, with c order and strides [channels*H*W, H*W, W, 1]
        //(b) Epsilons come from CNN layer above, with c order and strides [H*W, channels*H*W, W, 1] (i.e., due to permute)

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
        } else if (!Arrays.equals(new long[] {outH * outW, inDepth * outH * outW, outW, 1}, epsilon.stride())) {
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
                DynamicCustomOp op = DynamicCustomOp.builder("im2col")
                        .addIntegerArguments(kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                                ArrayUtil.fromBoolean(convolutionMode == ConvolutionMode.Same))
                        .addFloatingPointArguments(minValue())
                        .addInputs(input)
                        .addOutputs(col6dPermuted)
                        .build();
                Nd4j.getExecutioner().exec(op);

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
            default:
                throw new IllegalStateException("Unknown or unsupported pooling type: " + layerConf().getPoolingType()
                        + " " + layerId());
        }

        //Finally: we want the output strides for the epsilons to match the strides in the activations from the layer below
        //Assuming the layer below is a CNN layer (very likely) we want [H*W, channels*H*W, W, 1] instead of the standard
        // c-order [channels*H*W, H*W, W, 1] strides
        //To achieve this: [channels, miniBatch, H, W] in c order, then permute to [miniBatch, channels, H, W]
        //This gives us proper strides of 1 on the muli...
        INDArray tempEpsilon = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, new int[] {inDepth, miniBatch, inH, inW}, 'c');
        INDArray outEpsilon = tempEpsilon.permute(1, 0, 2, 3);
        Convolution.col2im(col6dPermuted, outEpsilon, strides[0], strides[1], pad[0], pad[1], inputHeight, inputWidth, dilation[0], dilation[1]);

        if (layerConf().getPoolingType() == PoolingType.AVG)
            outEpsilon.divi(ArrayUtil.prod(layerConf().getKernelSize()));

        outEpsilon = backpropDropOutIfPresent(outEpsilon);
        return new Pair<>(retGradient, outEpsilon);
    }

    private static double minValue(){
        switch (Nd4j.dataType()){
            case DOUBLE:
                return -Double.MAX_VALUE;
            case FLOAT:
                return -Float.MAX_VALUE;
            case HALF:
                return -65504.0;
            default:
                throw new IllegalStateException("Unexpected data type: " + Nd4j.dataType());
        }
    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        if (training && !dropoutApplied && layerConf().getIDropout() != null) {
            applyDropOutIfNecessary(true, workspaceMgr);
        }

        //Input validation: expect rank 4 matrix
        if (input.rank() != 4) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                            + " array as input to SubsamplingLayer with shape " + Arrays.toString(input.shape())
                            + ". Expected rank 4 array with shape [minibatchSize, channels, inputHeight, inputWidth]. "
                            + layerId());
        }

        // FIXME: int cast
        int miniBatch = (int) input.size(0);
        int inDepth = (int) input.size(1);
        int inH = (int) input.size(2);
        int inW = (int) input.size(3);

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

        if (helper != null && (helperCountFail == 0 || !layerConf().isCudnnAllowFallback())) {
            INDArray ret = null;
            try {
                ret = helper.activate(input, training, kernel, strides, pad, layerConf().getPoolingType(),
                        convolutionMode, dilation, workspaceMgr);
            } catch (Exception e){
                if(layerConf().isCudnnAllowFallback()){
                    helperCountFail++;
                    log.warn("CuDNN execution failed - falling back on built-in implementation",e);
                } else {
                    throw new RuntimeException(e);
                }
            }
            if (ret != null) {
                return ret;
            }
        }

        //Similar to convolution layer forward pass: do im2col, but permute so that pooling can be done with efficient strides...
        //Current im2col implementation expects input with shape [miniBatch,channels,kH,kW,outH,outW]

        INDArray output = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new int[]{miniBatch, inDepth, outH, outW}, 'c');

        LegacyPooling2D.Pooling2DType pt;
        double extra = 0.0;
        switch (layerConf().getPoolingType()){
            case MAX:
                pt = LegacyPooling2D.Pooling2DType.MAX;
                break;
            case AVG:
                pt = LegacyPooling2D.Pooling2DType.AVG;
                extra = 1.0;    //Divide by kH*kW not "number present" to match backward pass
                break;
            case PNORM:
                pt = LegacyPooling2D.Pooling2DType.PNORM;
                extra = layerConf().getPnorm();
                break;
            default:
                throw new UnsupportedOperationException("Not supported: " + layerConf().getPoolingType());
        }
        Op op = new LegacyPooling2D(input, kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], dilation[0], dilation[1],
                convolutionMode == ConvolutionMode.Same, pt, extra, output);
        Nd4j.getExecutioner().exec(op);

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
    public LayerHelper getHelper() {
        return helper;
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
    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {}

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
}
