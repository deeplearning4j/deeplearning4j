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

package org.deeplearning4j.nn.layers.convolution.subsampling;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.IsMax;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;


import java.util.*;


/**
 * Subsampling layer.
 *
 * Used for downsampling a convolution
 *
 * @author Adam Gibson
 */
public class SubsamplingLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.SubsamplingLayer> {
    private INDArray maxIndexes;

    public SubsamplingLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public SubsamplingLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


    @Override
    public double calcL2() {
        return 0;
    }

    @Override
    public double calcL1() {
        return 0;
    }

    @Override
    public Type type() {
        return Type.SUBSAMPLING;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        int miniBatch = input.size(0);
        int depth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] pad = layerConf().getPadding();

        int outH = Convolution.outSize(inH, kernel[0], strides[0], pad[0],false);
        int outW = Convolution.outSize(inW, kernel[1], strides[1], pad[1], false);

        //subsampling doesn't have weights and thus gradients are not calculated for this layer
        //only scale and reshape epsilon
        int inputHeight = input().size(-2);
        int inputWidth = input().size(-1);
        INDArray reshapeEpsilon, retE;
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
        if(epsilon.ordering() != 'c'){
            epsilon = epsilon.dup('c');
            cOrderStrides = true;
        }
        if(!cOrderStrides && Shape.strideDescendingCAscendingF(epsilon)){
            cOrderStrides = true;
        } else if(!Arrays.equals(new int[]{outH*outW, depth*outH*outW, outW, 1}, epsilon.stride())){
            //Unexpected/unusual strides, not either (a) or (b) cases above
            epsilon = epsilon.dup('c');
            cOrderStrides = true;
        }

        INDArray col6d;
        INDArray col6dPermuted;
        INDArray epsilon1d;
        if(cOrderStrides){
            //"Dense/Output layer above strides... i.e., standard c-order strides
            col6d = Nd4j.create(new int[]{miniBatch,depth,outH,outW,kernel[0],kernel[1]},'c');
            col6dPermuted = col6d.permute(0,1,4,5,2,3);
            epsilon1d = epsilon.reshape('c', ArrayUtil.prod(epsilon.length()), 1);  //zero copy reshape
        } else {
            //"CNN layer above" strides...
            col6d = Nd4j.create(new int[]{depth,miniBatch,outH,outW,kernel[0],kernel[1]},'c');
            col6dPermuted = col6d.permute(1,0,4,5,2,3);

            INDArray epsilonTemp = epsilon.permute(1,0,2,3);
            epsilon1d = epsilonTemp.reshape('c', new int[]{ArrayUtil.prod(epsilon.length()),1} );   //Should be a zero-copy reshape always
        }

        INDArray col2d = col6d.reshape('c',miniBatch*depth*outH*outW,kernel[0]*kernel[1]);


        switch(layerConf().getPoolingType()) {
            case MAX:
                //Execute im2col, then reshape to 2d. Note rows are in a different order for cOrderStrides true vs false cases
                Convolution.im2col(input, kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], false, col6dPermuted);
                INDArray isMax = Nd4j.getExecutioner().execAndReturn(new IsMax(col2d,1));
                isMax.muliColumnVector(epsilon1d);
                break;
            case AVG:
                //TODO: We could further optimize this by creating an uninitialized array, and doing a 'putiRowVector' operation
                // instead of a zero initialization + an addiRowVector op
                col2d.addiRowVector(epsilon1d);
                break;
            case NONE:
                return new Pair<>(retGradient, epsilon);
            default: throw new IllegalStateException("Unsupported pooling type");
        }

        //Finally: we want the output strides for the epsilons to match the strides in the activations from the layer below
        //Assuming the layer below is a CNN layer (very likely) we want [H*W, depth*H*W, W, 1] instead of the standard
        // c-order [depth*H*W, H*W, W, 1] strides
        //To achieve this: [depth, miniBatch, H, W] in c order, then permute to [miniBatch, depth, H, W]
        //This gives us proper strides of 1 on the muli...
        INDArray tempEpsilon = Nd4j.create(new int[]{depth,miniBatch,inH, inW},'c');
        INDArray outEpsilon = tempEpsilon.permute(1,0,2,3);
        Convolution.col2im(col6dPermuted, outEpsilon, strides[0], strides[1], pad[0], pad[1], inputHeight, inputWidth);

        if(layerConf().getPoolingType() == org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType.AVG)
            outEpsilon.divi(ArrayUtil.prod(layerConf().getKernelSize()));
        return new Pair<>(retGradient,outEpsilon);
    }


    @Override
    public INDArray activate(boolean training) {
        if(training && conf.getLayer().getDropOut() > 0) {
            this.dropoutMask = Dropout.applyDropout(input,conf.getLayer().getDropOut(),dropoutMask);
        }

        int miniBatch = input.size(0);
        int inDepth = input.size(1);
        int inH = input.size(2);
        int inW = input.size(3);

        int[] kernel = layerConf().getKernelSize();
        int[] strides = layerConf().getStride();
        int[] pad = layerConf().getPadding();

        int outH = Convolution.outSize(inH, kernel[0], strides[0], pad[0],false);
        int outW = Convolution.outSize(inW, kernel[1], strides[1], pad[1], false);

        //Similar to convolution layer forward pass: do im2col, but permute so that pooling can be done with efficient strides...
        //Current im2col implementation expects input with shape [miniBatch,depth,kH,kW,outH,outW]
        INDArray col = Nd4j.create(new int[]{miniBatch,inDepth,outH,outW,kernel[0],kernel[1]},'c');
        INDArray col2 = col.permute(0,1,4,5,2,3);
        Convolution.im2col(input, kernel[0], kernel[1], strides[0], strides[1], pad[0], pad[1], false, col2);

        //Reshape to 2d; should be zero-copy reshape due to permute above
        INDArray col2d = col.reshape('c',miniBatch*inDepth*outH*outW,kernel[0]*kernel[1]);

        INDArray reduced;
        switch(layerConf().getPoolingType()) {
            case AVG:
                reduced = col2d.mean(1);
                break;
            case MAX:
                reduced = col2d.max(1);
                break;
            case NONE:
                return input;
            default: throw new IllegalStateException("Unknown/not supported pooling type: " + layerConf().getPoolingType());
        }
        return reduced.reshape('c',miniBatch,inDepth,outH,outW);
    }

    @Override
    public Gradient error(INDArray input) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        throw new UnsupportedOperationException();
    }


    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray activationMean() {
        return null;
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void iterate(INDArray input) {
        throw new UnsupportedOperationException();
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

    }

    @Override
    public double score() {
        return 0;
    }

    @Override
    public void accumulateScore(double accum) { throw new UnsupportedOperationException(); }


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
