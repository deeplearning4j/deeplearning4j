package org.deeplearning4j.nn.layers.convolution;

import lombok.val;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.params.DeconvolutionParamInitializer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Arrays;

/**
 * 2D deconvolution layer implementation.
 *
 * Deconvolutions are also known as transpose convolutions or fractionally strided convolutions.
 * In essence, deconvolutions swap forward and backward pass with regular 2D convolutions.
 *
 * See the paper by Matt Zeiler for details:
 * http://www.matthewzeiler.com/wp-content/uploads/2017/07/cvpr2010.pdf
 *
 * For an intuitive guide to convolution arithmetic and shapes, see:
 * https://arxiv.org/abs/1603.07285v1
 *
 *
 * @author Max Pumperla
 */
public class Deconvolution2DLayer extends ConvolutionLayer {

    public Deconvolution2DLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public Deconvolution2DLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


    @Override
    void initializeHelper() {
        // no op
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

        INDArray weights = getParamWithNoise(DeconvolutionParamInitializer.WEIGHT_KEY, true, workspaceMgr);

        // FIXME: int cast
        int miniBatch = (int) input.size(0);
        int inH = (int) input.size(2);
        int inW = (int) input.size(3);

        int inDepth = (int) weights.size(0);

        int kH = (int) weights.size(2);
        int kW = (int) weights.size(3);

        int[] dilation = layerConf().getDilation();
        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] pad;
        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getDeconvolutionOutputSize(input, kernel, strides, null, convolutionMode, dilation);
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {inH, inW}, kernel, strides, dilation);
        } else {
            pad = layerConf().getPadding();
            outSize = ConvolutionUtils.getDeconvolutionOutputSize(input, kernel, strides, pad, convolutionMode, dilation);
        }

        INDArray biasGradView = gradientViews.get(DeconvolutionParamInitializer.BIAS_KEY);
        INDArray weightGradView = gradientViews.get(DeconvolutionParamInitializer.WEIGHT_KEY);

        INDArray outEps = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, new int[]{miniBatch, inDepth, inH, inW}, 'c');

        Integer sameMode = (convolutionMode == ConvolutionMode.Same) ? 1 : 0;

        int[] args = new int[] {
                kH, kW, strides[0], strides[1],
                pad[0], pad[1], dilation[0], dilation[1], sameMode
        };

        INDArray delta;
        IActivation afn = layerConf().getActivationFn();
        Pair<INDArray, INDArray> p = preOutput4d(true, true, workspaceMgr);
        delta = afn.backprop(p.getFirst(), epsilon).getFirst();

        INDArray[] opInputs;
        INDArray[] opOutputs;
        if(layerConf().hasBias()){
            INDArray bias = getParamWithNoise(DeconvolutionParamInitializer.BIAS_KEY, true, workspaceMgr);
            opInputs = new INDArray[]{input, weights, bias, delta};
            opOutputs = new INDArray[]{outEps, weightGradView, biasGradView};
        } else {
            opInputs = new INDArray[]{input, weights, delta};
            opOutputs = new INDArray[]{outEps, weightGradView};
        }
        CustomOp op = DynamicCustomOp.builder("deconv2d_bp")
                .addInputs(opInputs)
                .addIntegerArguments(args)
                .addOutputs(opOutputs)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(op);


        Gradient retGradient = new DefaultGradient();
        if(layerConf().hasBias()){
            retGradient.setGradientFor(DeconvolutionParamInitializer.BIAS_KEY, biasGradView);
        }
        retGradient.setGradientFor(DeconvolutionParamInitializer.WEIGHT_KEY, weightGradView, 'c');
        weightNoiseParams.clear();

        outEps = outEps.permute(1, 0, 2 , 3);

        return new Pair<>(retGradient, outEps);
    }

    @Override
    protected Pair<INDArray, INDArray> preOutput(boolean training , boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        if (convolutionMode == ConvolutionMode.Same) {
            throw new IllegalArgumentException("Border mode Same currently not supported.");
        }

        INDArray bias = getParamWithNoise(DeconvolutionParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray weights = getParamWithNoise(DeconvolutionParamInitializer.WEIGHT_KEY, training, workspaceMgr);

        //Input validation: expect rank 4 matrix
        if (input.rank() != 4) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to Deconvolution2D (layer name = " + layerName + ", layer index = "
                    + index + ") with shape " + Arrays.toString(input.shape()) + ". "
                    + "Expected rank 4 array with shape [minibatchSize, layerInputDepth, inputHeight, inputWidth]."
                    + (input.rank() == 2
                    ? " (Wrong input type (see InputType.convolutionalFlat()) or wrong data type?)"
                    : "")
                    + " " + layerId());
        }

        // FIXME: int cast
        int inDepth = (int) weights.size(0);
        int outDepth = (int) weights.size(1);

        if (input.size(1) != inDepth) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            throw new DL4JInvalidInputException("Cannot do forward pass in Deconvolution2D layer (layer name = " + layerName
                    + ", layer index = " + index + "): input array channels does not match CNN layer configuration"
                    + " (data input channels = " + input.size(1) + ", [minibatch,inputDepth,height,width]="
                    + Arrays.toString(input.shape()) + "; expected" + " input channels = " + inDepth + ") "
                    + layerId());
        }
        int kH = (int) weights.size(2);
        int kW = (int) weights.size(3);

        int[] dilation = layerConf().getDilation();
        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();

        int[] pad;
        int[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            // FIXME: int cast
            outSize = ConvolutionUtils.getDeconvolutionOutputSize(input, kernel, strides, null, convolutionMode, dilation); //Also performs validation
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new int[] {(int) input.size(2), (int) input.size(3)}, kernel,
                    strides, dilation );
        } else {
            pad = layerConf().getPadding();
            outSize = ConvolutionUtils.getDeconvolutionOutputSize(input, kernel, strides, pad, convolutionMode, dilation); //Also performs validation
        }

        int outH = outSize[0];
        int outW = outSize[1];


        val miniBatch = input.size(0);
        INDArray output = workspaceMgr.create(ArrayType.ACTIVATIONS, new long[]{miniBatch, outDepth, outH, outW}, 'c');

        int sameMode = (convolutionMode == ConvolutionMode.Same) ? 1 : 0;

        int[] args = new int[] {
                kH, kW, strides[0], strides[1],
                pad[0], pad[1], dilation[0], dilation[1], sameMode, 0   //Last arg: 0 for nchw
        };

        INDArray[] opInputs;
        if (layerConf().hasBias()) {
            opInputs = new INDArray[]{input, weights, bias};
        } else {
            opInputs = new INDArray[]{input, weights};
        }
        CustomOp op = DynamicCustomOp.builder("deconv2d")
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