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
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.loop.coordinatefunction.CoordinateFunction;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
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
        //subsampling doesn't have weights and thus gradients are not calculated for this layer
        //only scale and reshape epsilon
        int inputHeight = input().size(-2);
        int inputWidth = input().size(-1);
        INDArray reshapeEpsilon, retE, reshaped;
        Gradient retGradient = new DefaultGradient();

        switch(layerConf().getPoolingType()) {
            case MAX:
                int n = epsilon.size(0);
                int c = epsilon.size(1);
                int outH = epsilon.size(2);
                int outW = epsilon.size(3);
                //compute backwards kernel based on rearranging the given error
                retE = Nd4j.zeros(n, c, layerConf().getKernelSize()[0], layerConf().getKernelSize()[1], outH, outW);
                reshaped = retE.reshape(n, c, -1, outH, outW);
                reshapeEpsilon = Nd4j.rollAxis(reshaped,2);

                //Iterator<int[]> iter = new NdIndexIterator('c',true,n,c,outH,outW);
                final INDArray finalEps = epsilon;
                final INDArray reshapedEps = reshapeEpsilon;
                Shape.iterate(0, 4, new int[]{n, c, outH, outW}, new int[4], new CoordinateFunction() {
                    @Override
                    public void process(int[]... coord) {
                       try {
                           int[] i = coord[0];
                           double epsGet = finalEps.getDouble(i);
                           int idx = maxIndexes.getInt(i);
                           INDArray sliceToGetFrom = reshapedEps.get(NDArrayIndex.point(idx));
                           sliceToGetFrom.putScalar(i, epsGet);
                       }catch(Exception e) {
                           throw new IllegalStateException("Iterated to " + Arrays.toString(coord[0]) + " out of shape for indexes "
                                   + Arrays.toString(maxIndexes.shape()) + " and final eps shape " + Arrays.toString(finalEps.shape()));
                       }
                    }
                });



                reshapeEpsilon = Convolution.col2im(retE,layerConf().getStride(),layerConf().getPadding(),inputHeight, inputWidth);
                return new Pair<>(retGradient,reshapeEpsilon);
            case AVG:
                //compute reverse average error
                retE = epsilon.get(
                        NDArrayIndex.all()
                        , NDArrayIndex.all()
                        , NDArrayIndex.newAxis()
                        , NDArrayIndex.newAxis());
                reshapeEpsilon = Nd4j.tile(retE,1,1,layerConf().getKernelSize()[0],layerConf().getKernelSize()[1],1,1);
                reshapeEpsilon = Convolution.col2im(reshapeEpsilon, layerConf().getStride(), layerConf().getPadding(), inputHeight, inputWidth);
                reshapeEpsilon.divi(ArrayUtil.prod(layerConf().getKernelSize()));

                return new Pair<>(retGradient, reshapeEpsilon);
            case NONE:
                return new Pair<>(retGradient, epsilon);
            default: throw new IllegalStateException("Un supported pooling type");
        }
    }


    @Override
    public INDArray activate(boolean training) {
        INDArray pooled, ret;
        // n = num examples, c = num channels or depth
        int n, c, kh, kw, outWidth, outHeight;
        if(training && conf.getLayer().getDropOut() > 0) {
            this.dropoutMask = Dropout.applyDropout(input,conf.getLayer().getDropOut(),dropoutMask);
        }

        pooled = Convolution.im2col(input,layerConf().getKernelSize(),layerConf().getStride(),layerConf().getPadding());
        switch(layerConf().getPoolingType()) {
            case AVG:
                return pooled.mean(2,3);
            case MAX:
                n = pooled.size(0);
                c = pooled.size(1);
                kh = pooled.size(2);
                kw = pooled.size(3);
                outHeight = pooled.size(4);
                outWidth = pooled.size(5);
                ret = pooled.reshape(n, c, kh * kw, outHeight, outWidth);
                maxIndexes = Nd4j.argMax(ret, 2);
                return ret.max(2);
            case NONE:
                return input;
            default: throw new IllegalStateException("Pooling type not supported!");

        }
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
