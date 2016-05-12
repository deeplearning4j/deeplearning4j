/*
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

package org.deeplearning4j.nn.layers.convolution;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;


/**
 * Convolution layer
 *
 * @author Adam Gibson
 */
public class ConvolutionLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.ConvolutionLayer> {
//    protected INDArray col; // vectorized input

    public ConvolutionLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public ConvolutionLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


//    public void setCol(INDArray col) {this.col = col;}

    @Override
    public double calcL2() {
    	if(!conf.isUseRegularization() || conf.getLayer().getL2() <= 0.0 ) return 0.0;

        double l2Norm = getParam(ConvolutionParamInitializer.WEIGHT_KEY).norm2Number().doubleValue();
        return 0.5 * conf.getLayer().getL2() * l2Norm * l2Norm;
    }

    @Override
    public double calcL1() {
    	if(!conf.isUseRegularization() || conf.getLayer().getL1() <= 0.0 ) return 0.0;
        return conf.getLayer().getL1() * getParam(ConvolutionParamInitializer.WEIGHT_KEY).norm1Number().doubleValue();
    }

    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }

    public INDArray calculateDelta(INDArray epsilon) {
        INDArray z = preOutput(true);
        INDArray activationDerivative = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), z).derivative());
        if(!Arrays.equals(z.shape(),activationDerivative.shape()))
            throw new IllegalStateException("Shapes must be same");
        return epsilon.muli(activationDerivative);

    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        int inputHeight = input().size(-2);
        int inputWidth = input().size(-1);
        INDArray weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);

        // gy, Note: epsilon should be reshaped to a tensor when passed in
        INDArray delta = calculateDelta(epsilon);

        Gradient retGradient = new DefaultGradient();

        //gb = gy[0].sum(axis=(0, 2, 3))
        retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, delta.sum(0, 2, 3));

        INDArray col = null;

        // gW = np.tensordot(gy[0], col, ([0, 2, 3], [0, 4, 5]))
        INDArray weightGradient = Nd4j.tensorMmul(delta, col, new int[][] {{0, 2, 3},{0, 4, 5}});
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, weightGradient, 'c');

        //gcol = tensorMmul(W, gy[0], (0, 1))
        INDArray nextEpsilon = Nd4j.tensorMmul(weights, delta, new int[][] {{0}, {1}});

        nextEpsilon = Nd4j.rollAxis(nextEpsilon, 3);
        nextEpsilon = Convolution.col2im(nextEpsilon, layerConf().getStride(), layerConf().getPadding(), inputHeight, inputWidth);
        return new Pair<>(retGradient,nextEpsilon);
    }

    public INDArray preOutput(boolean training) {
        INDArray weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);
        INDArray bias = getParam(ConvolutionParamInitializer.BIAS_KEY);
        if(conf.isUseDropConnect() && training) {
            if (conf.getLayer().getDropOut() > 0) {
                weights = Dropout.applyDropConnect(this, ConvolutionParamInitializer.WEIGHT_KEY);
            }
        }

        int miniBatch = input.size(0);
        int inH = input.size(2);
        int inW = input.size(3);

        int outDepth = weights.size(0);
        int inDepth = weights.size(1);
        int kH = weights.size(2);
        int kW = weights.size(3);

        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] pad = layerConf().getPadding();

        int outH = Convolution.outSize(inH, kernel[0], strides[0], pad[0],false);
        int outW = Convolution.outSize(inW, kernel[1], strides[1], pad[1], false);

        //im2col in the required order: want [outW,outH,miniBatch,depthIn,kH,kW], but need to input [miniBatch,depth,kH,kW,outH,outW]
        // given the current im2col implementation
        //To get this: create an array of the order we want, permute it to the order required by im2col implementation, and then do im2col on that
        //to get old order from required order: permute(2,3,4,5,1,2)
        INDArray col = Nd4j.create(new int[]{outW,outH,miniBatch,inDepth,kH,kW},'c');
        INDArray col2 = col.permute(2,3,4,5,0,1);
        Convolution.im2col(input, kH, kW, strides[0], strides[1], pad[0], pad[1], false, col2);

        INDArray reshapedCol = Shape.newShapeNoCopy(col,new int[]{miniBatch*outH*outW, inDepth*kH*kW},false);
        if(reshapedCol == null) throw new RuntimeException("Could not reshape without copy");

        //Current order of weights: [depthOut,depthIn,kH,kW], c order
        //Permute to give [kW,kH,depthIn,depthOut], f order
        //Then zero-copy reshape to give [kW*kH*depthIn, depthOut]

        if(weights.ordering() != 'c') throw new RuntimeException("Weights not c order");
        INDArray permutedW = weights.permute(3,2,1,0);
        if(permutedW.ordering() != 'f') throw new RuntimeException("Not 'f' order after reshaping");
        INDArray reshapedW = Shape.newShapeNoCopy(permutedW,new int[]{kW*kH*inDepth,outDepth},true);

        if(reshapedW==null){
            throw new RuntimeException("Could not reshape weights");
        }
        if(reshapedW.ordering() != 'f') throw new RuntimeException("Not 'f' order after reshaping");

        //Do the MMUL; c and f orders in, c order out
        INDArray z = Nd4j.gemm(reshapedCol,reshapedW,false,false);

        //Expected output shape: [miniBatch*outH*outW,depthOut]
        if(z.ordering() != 'f') throw new RuntimeException();
        if(z.rows() != miniBatch*outH*outW) throw new RuntimeException();
        if(z.columns() != outDepth) throw new RuntimeException();

        //Now, reshape to [miniBatch,outH,outW,depthOut]
        z = z.reshape('f',outW,outH,miniBatch,outDepth);

        BroadcastOp op = new BroadcastAddOp(z,bias,z,3);
        Nd4j.getExecutioner().exec(op);

        //Output activations with shape [miniBath,outDepth,outH,outW];
        INDArray out = z.permute(2,3,1,0);
        return out;
    }

    @Override
    public INDArray activate(boolean training) {
        if(input == null)
            throw new IllegalArgumentException("No null input allowed");
        applyDropOutIfNecessary(training);

        INDArray z = preOutput(training);

        INDArray activation = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), z));
        return activation;
    }

    @Override
    public Layer transpose(){
        throw new UnsupportedOperationException("Not yet implemented");
    }


    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void fit(INDArray input) {}

    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray params(){
        //C order flattening, to match the gradient flattening order
        return Nd4j.toFlattened('c',params.values());
    }

    @Override
    public void setParams(INDArray params){
        //Override, as base layer does f order parameter flattening by default
        setParams(params,'c');
    }

}
