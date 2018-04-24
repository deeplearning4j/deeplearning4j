package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.Convolution3DParamInitializer;
import org.deeplearning4j.util.Convolution3DUtils;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

/**
 * 3D convolution layer implementation.
 *
 * @author Max Pumperla
 */
public class Convolution3DLayer extends ConvolutionLayer {

    public Convolution3DLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public Convolution3DLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


    @Override
    void initializeHelper() {
        // no op
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {

        if (input.rank() != 5) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to SubsamplingLayer with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 5 array with shape [minibatchSize, channels, "
                    + "inputHeight, inputWidth, inputDepth]. "
                    + layerId());
        }

        INDArray weights = getParamWithNoise(Convolution3DParamInitializer.WEIGHT_KEY, true);

        Convolution3D layerConfig = (Convolution3D) layerConf();

        boolean isNCDHW = layerConfig.getDataFormat() == Convolution3D.DataFormat.NCDHW;

        int miniBatch = input.size(0);
        int inD = isNCDHW ? input.size(2) : input.size(1);
        int inH = isNCDHW ? input.size(3) : input.size(2);
        int inW = isNCDHW ? input.size(4) : input.size(3);

        int outChannels = isNCDHW ? weights.size(0) : weights.size(4);

        int[] dilation = layerConfig.getDilation();
        int[] kernel = layerConfig.getKernelSize();
        int[] strides = layerConfig.getStride();
        int[] pad;
        int[] outSize;

        if (convolutionMode == ConvolutionMode.Same) {
            outSize = Convolution3DUtils.get3DOutputSize(
                    input, kernel, strides, null, convolutionMode, dilation, isNCDHW);
            pad = Convolution3DUtils.get3DSameModeTopLeftPadding(
                    outSize, new int[]{inD, inH, inW}, kernel, strides, dilation);
        } else {
            pad = layerConfig.getPadding();
            outSize = Convolution3DUtils.get3DOutputSize(
                    input, kernel, strides, pad, convolutionMode, dilation, isNCDHW);
        }
        int outD = outSize[0];
        int outH = outSize[1];
        int outW = outSize[2];

        INDArray weightGradView = gradientViews.get(Convolution3DParamInitializer.WEIGHT_KEY);

        INDArray outEpsilon = Nd4j.create(miniBatch * outChannels * outH * outW * outD);
        INDArray reshapedEpsilon;
        if (isNCDHW)
            reshapedEpsilon = outEpsilon.reshape('c', miniBatch, outChannels, outD, outH, outW);
        else
            reshapedEpsilon = outEpsilon.reshape('c', miniBatch, outD, outH, outW, outChannels);


        int[] intArgs = new int[]{
                kernel[0], kernel[1], kernel[2],
                strides[0], strides[1], strides[2],
                pad[0], pad[1], pad[2],
                dilation[0], dilation[1], dilation[2],
                convolutionMode == ConvolutionMode.Same ? 0 : 1,
                isNCDHW ? 1 : 0
        };

        INDArray delta;
        IActivation afn = layerConfig.getActivationFn();
        Pair<INDArray, INDArray> p = preOutput(true, true);
        delta = afn.backprop(p.getFirst(), epsilon).getFirst();

        INDArray bias;
        INDArray biasGradView = null;

        CustomOp op;
        if (layerConfig.hasBias()) {

            biasGradView = gradientViews.get(Convolution3DParamInitializer.BIAS_KEY);
            bias = getParamWithNoise(Convolution3DParamInitializer.BIAS_KEY, true);

            op = DynamicCustomOp.builder("conv3dnew_bp")
                    .addInputs(input, weights, bias, delta)
                    .addIntegerArguments(intArgs)
                    .addOutputs(reshapedEpsilon, weightGradView, biasGradView)
                    .callInplace(false)
                    .build();
        } else {
            op = DynamicCustomOp.builder("conv3dnew_bp")
                    .addInputs(input, weights, delta)
                    .addIntegerArguments(intArgs)
                    .addOutputs(reshapedEpsilon, weightGradView)
                    .callInplace(false)
                    .build();
        }
        Nd4j.getExecutioner().exec(op);

        Gradient retGradient = new DefaultGradient();
        if (layerConfig.hasBias()) {
            retGradient.setGradientFor(Convolution3DParamInitializer.BIAS_KEY, biasGradView);
        }
        retGradient.setGradientFor(Convolution3DParamInitializer.WEIGHT_KEY, weightGradView, 'c');
        weightNoiseParams.clear();

        return new Pair<>(retGradient, reshapedEpsilon);
    }


    @Override
    public INDArray preOutput(boolean training) {
        return preOutput(training, false).getFirst();
    }

    protected Pair<INDArray, INDArray> preOutput(boolean training, boolean forBackprop) {

        Convolution3D layerConfig = (Convolution3D) layerConf();

        ConvolutionMode mode = layerConfig.getConvolutionMode();
        boolean isNCDHW = layerConfig.getDataFormat() == Convolution3D.DataFormat.NCDHW;

        INDArray weights = getParamWithNoise(Convolution3DParamInitializer.WEIGHT_KEY, training);

        if (input.rank() != 5) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to Convolution3DLayer (layer name = " + layerName + ", layer index = "
                    + index + ") with shape " + Arrays.toString(input.shape()) + ". "
                    + "Expected rank 5 array with shape [minibatchSize, numChannels, inputHeight,"
                    + "inputWidth, inputDepth]."
                    + (input.rank() == 2
                    ? " (Wrong input type (see InputType.convolutionalFlat()) or wrong data type?)"
                    : "")
                    + " " + layerId());
        }


        int miniBatch = input.size(0);
        int inputChannels = isNCDHW ? input.size(1) : input.size(4);
        int inD = isNCDHW ? input.size(2) : input.size(1);
        int inH = isNCDHW ? input.size(3) : input.size(2);
        int inW = isNCDHW ? input.size(4) : input.size(3);

        int outWeightChannels = isNCDHW ? weights.size(0) : weights.size(4);
        int inWeightChannels = isNCDHW ? weights.size(1) : weights.size(3);

        if (inputChannels != inWeightChannels) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            throw new DL4JInvalidInputException("Cannot do forward pass in Convolution3D layer (layer name = "
                    + layerName
                    + ", layer index = " + index + "): number of input array channels does not match " +
                    "CNN layer configuration"
                    + " (data input channels = " + input.size(1)
                    + ", [minibatch, inputChannels, depth, height, width]="
                    + Arrays.toString(input.shape()) + "; expected" + " input channels = " + inWeightChannels + ") "
                    + layerId());
        }


        int[] kernel = layerConfig.getKernelSize();
        int[] dilation = layerConfig.getDilation();
        int[] strides = layerConfig.getStride();

        int[] pad;
        int[] outSize;
        if (mode == ConvolutionMode.Same) {
            outSize = Convolution3DUtils.get3DOutputSize(
                    input, kernel, strides, null, convolutionMode, dilation, isNCDHW);
            int[] inSize = new int[]{inD, inH, inW};
            pad = Convolution3DUtils.get3DSameModeTopLeftPadding(outSize,
                    inSize, kernel, strides, dilation);
        } else {
            pad = layerConfig.getPadding();
            outSize = Convolution3DUtils.get3DOutputSize(input, kernel, strides, pad, convolutionMode, dilation, isNCDHW);
        }
        int outD = outSize[0];
        int outH = outSize[1];
        int outW = outSize[2];

        INDArray output = Nd4j.create(miniBatch * outWeightChannels * outH * outW * outD);
        INDArray reshapedOutput;
        if (isNCDHW)
            reshapedOutput = output.reshape('c', miniBatch, outWeightChannels, outD, outH, outW);
        else
            reshapedOutput = output.reshape('c', miniBatch, outD, outH, outW, outWeightChannels);

        int[] intArgs = new int[]{
                kernel[0], kernel[1], kernel[2],
                strides[0], strides[1], strides[2],
                pad[0], pad[1], pad[2],
                dilation[0], dilation[1], dilation[2],
                mode == ConvolutionMode.Same ? 1 : 0,
                isNCDHW ? 0 : 1
        };

        CustomOp op;
        if (layerConfig.hasBias()) {
            INDArray bias = getParamWithNoise(Convolution3DParamInitializer.BIAS_KEY, training);

            op = DynamicCustomOp.builder("conv3dnew")
                    .addInputs(input, weights, bias)
                    .addIntegerArguments(intArgs)
                    .addOutputs(reshapedOutput)
                    .callInplace(false)
                    .build();
        } else {
            op = DynamicCustomOp.builder("conv3dnew")
                    .addInputs(input, weights)
                    .addIntegerArguments(intArgs)
                    .addOutputs(reshapedOutput)
                    .callInplace(false)
                    .build();
        }
        Nd4j.getExecutioner().exec(op);

        return new Pair<>(reshapedOutput, null);
    }
}