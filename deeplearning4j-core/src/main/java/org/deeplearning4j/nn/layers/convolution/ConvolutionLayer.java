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
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;


/**
 * Convolution layer
 *
 * @author Adam Gibson (original impl), Alex Black (current version)
 */
public class ConvolutionLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.ConvolutionLayer> {
    public ConvolutionLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public ConvolutionLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

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

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        INDArray weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);

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


        INDArray biasGradView = gradientViews.get(ConvolutionParamInitializer.BIAS_KEY);
        INDArray weightGradView = gradientViews.get(ConvolutionParamInitializer.WEIGHT_KEY);    //4d, c order. Shape: [outDepth,inDepth,kH,kW]
        INDArray weightGradView2df = Shape.newShapeNoCopy(weightGradView, new int[]{outDepth,inDepth*kH*kW}, false).transpose();



        INDArray delta;
        String afn = conf.getLayer().getActivationFunction();

        if("identity".equals(afn)){
            delta = epsilon;    //avoid doing .muli with 1s
        } else {
            INDArray sigmaPrimeZ = preOutput(true);
            Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(
                    afn, sigmaPrimeZ, conf.getExtraArgs()).derivative());
            delta = sigmaPrimeZ.muli(epsilon);  //Current shape: [miniBatch,outD,outH,outW]
        }

        delta = delta.permute(1,0,2,3); //To shape: [outDepth,miniBatch,outH,outW]

        //Note: due to the permute in preOut, and the fact that we essentially do a preOut.muli(epsilon), this reshape
        // should be zero-copy; only possible exception being sometimes with the "identity" activation case
        INDArray delta2d = delta.reshape('c',new int[]{outDepth,miniBatch*outH*outW});    //Shape.newShapeNoCopy(delta,new int[]{outDepth,miniBatch*outH*outW},false);

        //Do im2col, but with order [miniB,outH,outW,depthIn,kH,kW]; but need to input [miniBatch,depth,kH,kW,outH,outW] given the current im2col implementation
        //To get this: create an array of the order we want, permute it to the order required by im2col implementation, and then do im2col on that
        //to get old order from required order: permute(0,3,4,5,1,2)
        INDArray col = Nd4j.createUninitialized(new int[]{miniBatch,outH,outW,inDepth,kH,kW},'c');
        INDArray col2 = col.permute(0,3,4,5,1,2);
        Convolution.im2col(input, kH, kW, strides[0], strides[1], pad[0], pad[1], false, col2);

        //Shape im2col to 2d. Due to the permuting above, this should be a zero-copy reshape
        INDArray im2col2d = col.reshape('c', miniBatch*outH*outW, inDepth*kH*kW);

        //Calculate weight gradients, using cc->c mmul.
        //weightGradView2df is f order, but this is because it's transposed from c order
        //Here, we are using the fact that AB = (B^T A^T)^T; output here (post transpose) is in c order, not usual f order
        Nd4j.gemm(im2col2d,delta2d,weightGradView2df,true,true,1.0,0.0);

        //Flatten 4d weights to 2d... this again is a zero-copy op (unless weights are not originally in c order for some reason)
        INDArray wPermuted = weights.permute(3,2,1,0);  //Start with c order weights, switch order to f order
        INDArray w2d = wPermuted.reshape('f',inDepth*kH*kW, outDepth);

        //Calculate epsilons for layer below, in 2d format (note: this is in 'image patch' format before col2im reduction)
        //Note: cc -> f mmul here, then reshape to 6d in f order
        INDArray epsNext2d = w2d.mmul(delta2d);
        INDArray eps6d = Shape.newShapeNoCopy(epsNext2d,new int[]{kW,kH,inDepth,outW,outH,miniBatch}, true);

        //Calculate epsilonNext by doing im2col reduction.
        //Current col2im implementation expects input with order: [miniBatch,depth,kH,kW,outH,outW]
        //currently have [kH,kW,inDepth,outW,outH,miniBatch] -> permute first
        eps6d = eps6d.permute(5,2,1,0,4,3);
        INDArray epsNextOrig = Nd4j.create(new int[]{inDepth,miniBatch,inH,inW},'c');
        //Note: we are execute col2im in a way that the output array should be used in a stride 1 muli in the layer below... (same strides as zs/activations)
        INDArray epsNext = epsNextOrig.permute(1,0,2,3);
        Convolution.col2im(eps6d, epsNext, strides[0], strides[1], pad[0], pad[1], inH, inW);

        Gradient retGradient = new DefaultGradient();
        INDArray biasGradTemp = delta2d.sum(1);
        biasGradView.assign(biasGradTemp); //TODO do this properly, without the assign

        retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, biasGradView);
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, weightGradView, 'c');

        return new Pair<>(retGradient,epsNext);
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

        //im2col in the required order: want [outW,outH,miniBatch,depthIn,kH,kW], but need to input [miniBatch,depth,kH,kW,outH,outW] given the current im2col implementation
        //To get this: create an array of the order we want, permute it to the order required by im2col implementation, and then do im2col on that
        //to get old order from required order: permute(0,3,4,5,1,2)
        //Post reshaping: rows are such that minibatch varies slowest, outW fastest as we step through the rows post-reshape
        INDArray col = Nd4j.createUninitialized(new int[]{miniBatch,outH,outW,inDepth,kH,kW},'c');
        INDArray col2 = col.permute(0,3,4,5,1,2);
        Convolution.im2col(input, kH, kW, strides[0], strides[1], pad[0], pad[1], false, col2);

        INDArray reshapedCol = Shape.newShapeNoCopy(col,new int[]{miniBatch*outH*outW, inDepth*kH*kW},false);

        //Current order of weights: [depthOut,depthIn,kH,kW], c order
        //Permute to give [kW,kH,depthIn,depthOut], f order
        //Reshape to give [kW*kH*depthIn, depthOut]. This should always be zero-copy reshape, unless weights aren't in c order for some reason
        INDArray permutedW = weights.permute(3,2,1,0);
        INDArray reshapedW = permutedW.reshape('f',kW*kH*inDepth,outDepth);

        //Do the MMUL; c and f orders in, f order out. output shape: [miniBatch*outH*outW,depthOut]
        INDArray z = reshapedCol.mmul(reshapedW);

        //Add biases, before reshaping. Note that biases are [1,depthOut] and currently z is [miniBatch*outH*outW,depthOut] -> addiRowVector
        z.addiRowVector(bias);

        //Now, reshape to [outW,outH,miniBatch,outDepth], and permute to have correct output order: [miniBath,outDepth,outH,outW];
        z = Shape.newShapeNoCopy(z,new int[]{outW,outH,miniBatch,outDepth},true);
        return z.permute(2,3,1,0);
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
