package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.params.SeparableConvolutionParamInitializer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

/**
 * 2D Separable convolution layer implementation
 *
 * Separable convolutions split a regular convolution operation into two
 * simpler operations, which are usually computationally more efficient.
 *
 * The first step in a separable convolution is a depth-wise convolution, which
 * operates on each of the input maps separately. A depth multiplier is used to
 * specify the number of outputs per input map in this step. This convolution
 * is carried out with the specified kernel sizes, stride and padding values.
 *
 * The second step is a point-wise operation, in which the intermediary outputs
 * of the depth-wise convolution are mapped to the desired number of feature
 * maps, by using a 1x1 convolution.
 *
 * The result of chaining these two operations will result in a tensor of the
 * same shape as that for a standard conv2d operation.
 *
 * @author Max Pumperla
 */
public class SeparableConvolution2DLayer extends ConvolutionLayer {

    public SeparableConvolution2DLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public SeparableConvolution2DLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {

        if (input.rank() != 4) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to SubsamplingLayer with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 4 array with shape [minibatchSize, depth, inputHeight, inputWidth]. "
                    + layerId());
        }
        INDArray bias;
        INDArray depthWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, true);
        INDArray pointWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY, true);


        int miniBatch = input.size(0);
        int inH = input.size(2);
        int inW = input.size(3);

        int inDepth = depthWiseWeights.size(1);
        int kH = depthWiseWeights.size(2);
        int kW = depthWiseWeights.size(3);

        int[] dilation = layerConf().getDilation();
        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] pad;
        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(
                    input, kernel, strides, null, convolutionMode, dilation); //Also performs validation
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {inH, inW}, kernel, strides, dilation);
        } else {
            pad = layerConf().getPadding();
            outSize = ConvolutionUtils.getOutputSize(
                    input, kernel, strides, pad, convolutionMode, dilation); //Also performs validation
        }

        int outH = outSize[0];
        int outW = outSize[1];

        INDArray biasGradView = gradientViews.get(SeparableConvolutionParamInitializer.BIAS_KEY);
        INDArray depthWiseweightGradView = gradientViews.get(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY);
        INDArray pointWiseweightGradView = gradientViews.get(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY);

        INDArray outEpsilon = Nd4j.create(miniBatch * inDepth * inH * inW);
        INDArray reshapedEpsilon = outEpsilon.reshape('c', miniBatch, inDepth, inH, inW);

        Integer sameMode = (convolutionMode == ConvolutionMode.Same) ? 1 : 0;

        int[] args = new int[] {
                kH, kW, strides[0], strides[1],
                pad[0], pad[1], dilation[0], dilation[1], sameMode
        };

        INDArray delta;
        IActivation afn = layerConf().getActivationFn();
        Pair<INDArray, INDArray> p = preOutput4d(true, true);
        delta = afn.backprop(p.getFirst(), epsilon).getFirst();

        CustomOp op;
        if(layerConf().hasBias()){
            bias = getParamWithNoise(SeparableConvolutionParamInitializer.BIAS_KEY, true);

            op = DynamicCustomOp.builder("sconv2d_bp")
                    .addInputs(input, delta, depthWiseWeights, pointWiseWeights, bias)
                    .addIntegerArguments(args)
                    .addOutputs(reshapedEpsilon, depthWiseweightGradView, pointWiseweightGradView, biasGradView)
                    .callInplace(false)
                    .build();
        } else {
            op = DynamicCustomOp.builder("sconv2d_bp")
                    .addInputs(input, delta, depthWiseWeights, pointWiseWeights)
                    .addIntegerArguments(args)
                    .addOutputs(reshapedEpsilon, depthWiseweightGradView, pointWiseweightGradView)
                    .callInplace(false)
                    .build();
        }
        Nd4j.getExecutioner().exec(op);

        Gradient retGradient = new DefaultGradient();
        if(layerConf().hasBias()){
            retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, biasGradView);
        }
        retGradient.setGradientFor(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, depthWiseweightGradView, 'c');
        retGradient.setGradientFor(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY, pointWiseweightGradView, 'c');

        weightNoiseParams.clear();

        return new Pair<>(retGradient, reshapedEpsilon);
    }


    @Override
    public INDArray preOutput(boolean training) {
        return preOutput(training, false).getFirst();
    }

    protected Pair<INDArray, INDArray> preOutput(boolean training , boolean forBackprop) {

        INDArray bias = getParamWithNoise(SeparableConvolutionParamInitializer.BIAS_KEY, training);
        INDArray depthWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, training);
        INDArray pointWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY, training);

        if (input.rank() != 4) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to SeparableConvolution2D (layer name = " + layerName + ", layer index = "
                    + index + ") with shape " + Arrays.toString(input.shape()) + ". "
                    + "Expected rank 4 array with shape [minibatchSize, layerInputDepth, inputHeight, inputWidth]."
                    + (input.rank() == 2
                    ? " (Wrong input type (see InputType.convolutionalFlat()) or wrong data type?)"
                    : "")
                    + " " + layerId());
        }

        int inDepth = depthWiseWeights.size(1);
        int outDepth = pointWiseWeights.size(0);

        if (input.size(1) != inDepth) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            throw new DL4JInvalidInputException("Cannot do forward pass in SeparableConvolution2D layer (layer name = " + layerName
                    + ", layer index = " + index + "): input array depth does not match CNN layer configuration"
                    + " (data input depth = " + input.size(1) + ", [minibatch,inputDepth,height,width]="
                    + Arrays.toString(input.shape()) + "; expected" + " input depth = " + inDepth + ") "
                    + layerId());
        }
        int kH = depthWiseWeights.size(2);
        int kW = depthWiseWeights.size(3);

        int[] dilation = layerConf().getDilation();
        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();

        int[] pad;
        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode, dilation); //Also performs validation
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {input.size(2), input.size(3)}, kernel,
                    strides, dilation );
        } else {
            pad = layerConf().getPadding();
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode, dilation); //Also performs validation
        }

        int outH = outSize[0];
        int outW = outSize[1];


        if (helper != null) {
            if (preOutput != null && forBackprop) {
                return new Pair<>(preOutput, null);
            }

            //For no-bias convolutional layers: use an empty (all 0s) value for biases
            if(!hasBias()){
                if(dummyBias == null){
                    try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                        dummyBias = Nd4j.create(1, layerConf().getNOut());
                    }
                }
                bias = dummyBias;
            }

            INDArray ret = helper.preOutput(input, depthWiseWeights, bias, kernel, strides, pad, layerConf().getCudnnAlgoMode(),
                    layerConf().getCudnnFwdAlgo(), convolutionMode, dilation);
            if (ret != null) {
                return new Pair<>(ret, null);
            }
        }

        int miniBatch = input.size(0);
        INDArray output = Nd4j.create(miniBatch * outDepth * outH * outW);
        INDArray reshapedOutput = output.reshape('c', miniBatch, outDepth, outH, outW);

        Integer sameMode = (convolutionMode == ConvolutionMode.Same) ? 1 : 0;

        int[] args = new int[] {
                kH, kW, strides[0], strides[1],
                pad[0], pad[1], dilation[0], dilation[1], sameMode
        };

        CustomOp op;
        if (layerConf().hasBias()) {
            op = DynamicCustomOp.builder("sconv2d")
                .addInputs(input, depthWiseWeights, pointWiseWeights, bias)
                .addIntegerArguments(args)
                .addOutputs(reshapedOutput)
                .callInplace(false)
                .build();
        } else {
            op = DynamicCustomOp.builder("sconv2d")
                .addInputs(input, depthWiseWeights, pointWiseWeights)
                .addIntegerArguments(args)
                .addOutputs(reshapedOutput)
                .callInplace(false)
                .build();
        }
        Nd4j.getExecutioner().exec(op);

        return new Pair<>(reshapedOutput, null);
    }

    @Override
    public INDArray activate(boolean training) {
        if (input == null) {
            throw new IllegalArgumentException("Cannot perform forward pass with null input " + layerId());
        }

        if (cacheMode == null)
            cacheMode = CacheMode.NONE;

        applyDropOutIfNecessary(training);

        INDArray z = preOutput(training);

        // we do cache only if cache workspace exists. Skip otherwise
        if (training && cacheMode != CacheMode.NONE
                && Nd4j.getWorkspaceManager().checkIfWorkspaceExists(ComputationGraph.workspaceCache)) {
            try (MemoryWorkspace wsB = Nd4j.getWorkspaceManager()
                    .getWorkspaceForCurrentThread(ComputationGraph.workspaceCache).notifyScopeBorrowed()) {
                preOutput = z.unsafeDuplication();
            }
        }

        //String afn = conf.getLayer().getActivationFunction();
        IActivation afn = layerConf().getActivationFn();

        if (helper != null && Shape.strideDescendingCAscendingF(z)) {
            INDArray ret = helper.activate(z, layerConf().getActivationFn());
            if (ret != null) {
                return ret;
            }
        }

        INDArray activation = afn.getActivation(z, training);
        return activation;
    }
}
