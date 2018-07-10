package org.deeplearning4j.nn.layers.local;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LocallyConnected2D;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.convolution.ConvolutionLayer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.accum.Mmul;
import org.nd4j.linalg.api.ops.impl.shape.Unstack;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * A 2D locally connected layer computes a convolution with unshared weights. In a regular
 * convolution operation for each input filter there is one kernel that moves over patches
 * of the filter. In a locally connected layer, there is a separate kernel for each patch.
 *
 * @author Max Pumperla
 */
public class LocallyConnected2DLayer extends ConvolutionLayer {

    public LocallyConnected2DLayer(NeuralNetConfiguration conf) {
        super(conf);
    }


    public LocallyConnected2DLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


    /**
     * Pre-output method
     *
     * @param training    Train or test time (impacts dropout)
     * @param forBackprop If true: return the im2col2d array for re-use during backprop. False: return null for second
     *                    pair entry. Note that it may still be null in the case of CuDNN and the like.
     * @return            Pair of arrays: preOutput (activations) and optionally the im2col2d array
     */
    @Override
    protected Pair<INDArray, INDArray> preOutput(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        validateInputRank();

        INDArray bias = getParamWithNoise(ConvolutionParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray weights = getParamWithNoise(ConvolutionParamInitializer.WEIGHT_KEY, training, workspaceMgr);

        LocallyConnected2D layerConf = (LocallyConnected2D) layerConf();


        int miniBatch = (int) input.size(0);
        int outDepth = (int) layerConf.getNOut();
        int inDepth = (int) layerConf.getNIn();
        validateInputDepth(inDepth);

        int kH = layerConf.getKernelSize()[0];
        int kW = layerConf.getKernelSize()[1];

        int featureDim = inDepth * kH * kW;

        int sH = layerConf.getStride()[0];
        int sW = layerConf.getStride()[1];

        int outH = layerConf.getOutputSize()[0];
        int outW = layerConf.getOutputSize()[1];

        INDArray[] inputArray = new INDArray[outH * outW];
        for (int i = 0; i < outH; i++) { // iterate over all output patches
            for (int j = 0; j < outW; j++) {
                inputArray[i * outH + j] = input.get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(i * sH, i * sH + kH),
                        NDArrayIndex.interval(j * sW, j * sW + kW))
                        .reshape(1, miniBatch, featureDim);
            }
        }
        INDArray concatOutput = Nd4j.create(outH * outW, miniBatch, featureDim);
        DynamicCustomOp op = DynamicCustomOp.builder("concat")
                .addInputs(inputArray).addOutputs(concatOutput).addIntegerArguments(0).build();
        Nd4j.getExecutioner().exec(op);

        System.out.println(Arrays.toString(concatOutput.shape()));
        System.out.println(Arrays.toString(weights.shape()));



        INDArray mmulResult = Nd4j.create(new int[] {outH * outW, miniBatch, outDepth});
        MMulTranspose mMulTranspose = MMulTranspose.builder()
                .transposeB(false)
                .a(concatOutput)
                .b(weights)
                .build();
        Nd4j.getExecutioner().exec(new Mmul(concatOutput, weights, mmulResult, mMulTranspose));

        System.out.println(Arrays.toString(mmulResult.shape()));


        INDArray z = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, new long[]{10, 16, 6, 6});

        // TODO: permute & reshape input to (oH * oW, mb, kH * kW * nIn)
        // TODO kernel is of shape (oH * oW, kH * kW * nIn, nOut)
        // TODO: compute batched mmul, resulting in output shape (oH * oW, mb, nOut)
        // TODO: permute output to (mb, nOut, oH * oW)
        // TODO: reshape output to expected (mb, nOut, oH, oW)

        return new Pair<>(z, null);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        INDArray weights = getParamWithNoise(ConvolutionParamInitializer.WEIGHT_KEY, true, workspaceMgr);

        LocallyConnected2D layerConf = (LocallyConnected2D) layerConf();

        int miniBatch = (int) input.size(0);
        int inH = (int) input.size(2);
        int inW = (int) input.size(3);

        int outDepth = (int) layerConf.getNOut();
        int inDepth = (int) layerConf.getNIn();
        int kH = layerConf.getKernelSize()[0];
        int kW = layerConf.getKernelSize()[1];

        int outH = layerConf.getOutputSize()[0];
        int outW = layerConf.getOutputSize()[1];

        // TODO: compute backward pass, i.e. first compute backprop for activation function
        // TODO: then reshape epsilon from (mb, nOut, oH, oW) to (mb, nOut, oH * oW)
        // TODO: permute to (oH * oW, mb, nOut)
        // TODO: compute batch mmul to get gradients for kernels and output epsilons
        // TODO: set gradient updates (reshape first)
        // TODO: return next epsilon (reshape first) and gradients

        INDArray biasGradView = gradientViews.get(ConvolutionParamInitializer.BIAS_KEY);
        INDArray weightGradView = gradientViews.get(ConvolutionParamInitializer.WEIGHT_KEY);

        INDArray delta;
        IActivation afn = layerConf().getActivationFn();

        Pair<INDArray, INDArray> p = preOutput4d(true, true, workspaceMgr);
        delta = afn.backprop(p.getFirst(), epsilon).getFirst();

        Gradient retGradient = new DefaultGradient();

        weightNoiseParams.clear();

        INDArray epsNext = Nd4j.create(42);

        epsNext = backpropDropOutIfPresent(epsNext);
        return new Pair<>(retGradient, epsNext);
    }


}
