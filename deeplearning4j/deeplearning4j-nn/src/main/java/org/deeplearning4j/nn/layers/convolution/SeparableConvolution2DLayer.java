package org.deeplearning4j.nn.layers.convolution;

import lombok.val;
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
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Arrays;

/**
 * 2D Separable convolution layer implementation
 *
 * Separable convolutions split a regular convolution operation into two
 * simpler operations, which are usually computationally more efficient.
 *
 * The first step in a separable convolution is a channels-wise convolution, which
 * operates on each of the input maps separately. A channels multiplier is used to
 * specify the number of outputs per input map in this step. This convolution
 * is carried out with the specified kernel sizes, stride and padding values.
 *
 * The second step is a point-wise operation, in which the intermediary outputs
 * of the channels-wise convolution are mapped to the desired number of feature
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
    void initializeHelper(){
        //No op - no separable conv implementation in cudnn
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if (input.rank() != 4) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to SubsamplingLayer with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 4 array with shape [minibatchSize, channels, inputHeight, inputWidth]. "
                    + layerId());
        }
        INDArray bias;
        INDArray depthWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, true, workspaceMgr);
        INDArray pointWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY, true, workspaceMgr);


        // FIXME: int cast
        int miniBatch = (int) input.size(0);
        int inH = (int) input.size(2);
        int inW = (int) input.size(3);

        int inDepth = (int) depthWiseWeights.size(1);
        int kH = (int) depthWiseWeights.size(2);
        int kW = (int) depthWiseWeights.size(3);

        int[] dilation = layerConf().getDilation();
        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] pad;
        if (convolutionMode == ConvolutionMode.Same) {
            int[] outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode, dilation); //Also performs validation
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {inH, inW}, kernel, strides, dilation);
        } else {
            pad = layerConf().getPadding();
            ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode, dilation); //Also performs validation
        }

        INDArray biasGradView = gradientViews.get(SeparableConvolutionParamInitializer.BIAS_KEY);
        INDArray depthWiseweightGradView = gradientViews.get(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY);
        INDArray pointWiseweightGradView = gradientViews.get(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY);

        INDArray outEpsilon = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, new int[]{miniBatch, inDepth, inH, inW}, 'c');

        Integer sameMode = (convolutionMode == ConvolutionMode.Same) ? 1 : 0;

        int[] args = new int[] {
                kH, kW, strides[0], strides[1],
                pad[0], pad[1], dilation[0], dilation[1], sameMode
        };

        INDArray delta;
        IActivation afn = layerConf().getActivationFn();
        Pair<INDArray, INDArray> p = preOutput4d(true, true, workspaceMgr);
        delta = afn.backprop(p.getFirst(), epsilon).getFirst();

        CustomOp op;
        if(layerConf().hasBias()){
            bias = getParamWithNoise(SeparableConvolutionParamInitializer.BIAS_KEY, true, workspaceMgr);

            op = DynamicCustomOp.builder("sconv2d_bp")
                    .addInputs(input, delta, depthWiseWeights, pointWiseWeights, bias)
                    .addIntegerArguments(args)
                    .addOutputs(outEpsilon, depthWiseweightGradView, pointWiseweightGradView, biasGradView)
                    .callInplace(false)
                    .build();
        } else {
            op = DynamicCustomOp.builder("sconv2d_bp")
                    .addInputs(input, delta, depthWiseWeights, pointWiseWeights)
                    .addIntegerArguments(args)
                    .addOutputs(outEpsilon, depthWiseweightGradView, pointWiseweightGradView)
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

        outEpsilon = backpropDropOutIfPresent(outEpsilon);
        return new Pair<>(retGradient, outEpsilon);
    }

    @Override
    protected Pair<INDArray, INDArray> preOutput(boolean training , boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        INDArray bias = getParamWithNoise(SeparableConvolutionParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray depthWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, training, workspaceMgr);
        INDArray pointWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY, training, workspaceMgr);

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

        // FIXME: int cast
        int inDepth = (int) depthWiseWeights.size(1);
        int outDepth = (int) pointWiseWeights.size(0);

        if (input.size(1) != inDepth) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            throw new DL4JInvalidInputException("Cannot do forward pass in SeparableConvolution2D layer (layer name = " + layerName
                    + ", layer index = " + index + "): input array channels does not match CNN layer configuration"
                    + " (data input channels = " + input.size(1) + ", [minibatch,inputDepth,height,width]="
                    + Arrays.toString(input.shape()) + "; expected" + " input channels = " + inDepth + ") "
                    + layerId());
        }
        int kH = (int) depthWiseWeights.size(2);
        int kW = (int) depthWiseWeights.size(3);

        int[] dilation = layerConf().getDilation();
        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();

        int[] pad;
        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode, dilation); //Also performs validation

            // FIXME: int cast
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {(int) input.size(2), (int) input.size(3)}, kernel,
                    strides, dilation );
        } else {
            pad = layerConf().getPadding();
            outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode, dilation); //Also performs validation
        }

        int outH = outSize[0];
        int outW = outSize[1];

        val miniBatch = input.size(0);
        INDArray output = workspaceMgr.create(ArrayType.ACTIVATIONS, new long[]{miniBatch, outDepth, outH, outW}, 'c');

        Integer sameMode = (convolutionMode == ConvolutionMode.Same) ? 1 : 0;

        int[] args = new int[] {
                kH, kW, strides[0], strides[1],
                pad[0], pad[1], dilation[0], dilation[1], sameMode
        };

        INDArray[] opInputs;
        if (layerConf().hasBias()) {
            opInputs = new INDArray[]{input, depthWiseWeights, pointWiseWeights, bias};
        } else {
            opInputs = new INDArray[]{input, depthWiseWeights, pointWiseWeights};

        }
        CustomOp op = DynamicCustomOp.builder("sconv2d")
                .addInputs(opInputs)
                .addIntegerArguments(args)
                .addOutputs(output)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(op);

        return new Pair<>(output, null);
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);

        if (cacheMode == null)
            cacheMode = CacheMode.NONE;

        applyDropOutIfNecessary(training, workspaceMgr);

        INDArray z = preOutput(training, false, workspaceMgr).getFirst();

        //String afn = conf.getLayer().getActivationFunction();
        IActivation afn = layerConf().getActivationFn();

        INDArray activation = afn.getActivation(z, training);
        return activation;
    }
}
