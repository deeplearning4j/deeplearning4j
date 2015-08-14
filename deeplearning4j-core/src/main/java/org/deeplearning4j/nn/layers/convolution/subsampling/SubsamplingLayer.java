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

import com.google.common.primitives.Ints;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
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
public class SubsamplingLayer extends BaseLayer {
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
        return Type.CONVOLUTIONAL;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, Gradient gradient, Layer layer) {
        // TODO assign this.gradient to the gradient from here?
        //subsampling doesn't have weights and thus gradients are not calculated
        //only scale and reshape epsilon and gradient
        int n = epsilon.size(0);
        int c = epsilon.size(1);
        int outH = epsilon.size(2);
        int outW = epsilon.size(3);
        INDArray reshapeEpsilon, reshapeGradient, retE, retG;
        int inputHeight = input().size(-2);
        int inputWidth = input().size(-1);
        Gradient retGradient = new DefaultGradient();
        retGradient.gradientForVariable().put(ConvolutionParamInitializer.BIAS_KEY, gradient.getGradientFor("b"));

        switch(conf.getPoolingType()) {
            case MAX:
                //compute backwards kernel based on rearranging the given error
                retE = Nd4j.zeros(n, c, conf.getKernelSize()[0], conf.getKernelSize()[1], outH, outW);
                reshapeEpsilon = Nd4j.rollAxis(retE.reshape(n,c,-1,outH,outW),2);
                retG = Nd4j.zeros(n, c, conf.getKernelSize()[0], conf.getKernelSize()[1], outH, outW);
                reshapeGradient = Nd4j.rollAxis(retG.reshape(n,c,-1,outH,outW),2);

                Iterator<int[]> iter = new NdIndexIterator(n,c,outH,outW);
                while(iter.hasNext()) {
                    int[] next = iter.next();
                    INDArrayIndex[] indexes = NDArrayIndex.indexesFor(next);
                    reshapeEpsilon.get(indexes).put(indexes, epsilon.get(indexes));
                    reshapeGradient.get(indexes).put(indexes, gradient.getGradientFor("W").get(indexes));
                }
                reshapeEpsilon = Convolution.col2im(reshapeEpsilon,conf.getStride(),conf.getPadding(),inputHeight, inputWidth);
                retGradient.gradientForVariable().put(ConvolutionParamInitializer.WEIGHT_KEY, reshapeGradient);
                return new Pair<>(retGradient,reshapeEpsilon);
            case AVG:
                //compute reverse average error
                retE = epsilon.get(
                        NDArrayIndex.all()
                        , NDArrayIndex.all()
                        , NDArrayIndex.newAxis()
                        , NDArrayIndex.newAxis()
                        , NDArrayIndex.newAxis());
                reshapeEpsilon = Nd4j.tile(retE,1,1,conf.getKernelSize()[0],conf.getKernelSize()[1],1,1);

                reshapeEpsilon = Convolution.col2im(reshapeEpsilon, conf.getStride(), conf.getPadding(), inputHeight, inputWidth);
                reshapeEpsilon.divi(ArrayUtil.prod(conf.getKernelSize()));

                retG = epsilon.get(
                        NDArrayIndex.all()
                        , NDArrayIndex.all()
                        , NDArrayIndex.newAxis()
                        , NDArrayIndex.newAxis()
                        ,NDArrayIndex.newAxis());
                reshapeGradient = Nd4j.tile(retG,1,1,conf.getKernelSize()[0],conf.getKernelSize()[1],1,1);

                reshapeGradient = Convolution.col2im(reshapeGradient, conf.getStride(), conf.getPadding(), inputHeight, inputWidth);
                reshapeGradient.divi(ArrayUtil.prod(conf.getKernelSize()));

                retGradient.gradientForVariable().put(ConvolutionParamInitializer.WEIGHT_KEY, reshapeGradient);
                return new Pair<>(retGradient, reshapeEpsilon);
            case NONE:
                return new Pair<>(gradient, epsilon);
            default: throw new IllegalStateException("Un supported pooling type");
        }

    }


    @Override
    public INDArray activate(boolean training) {
        INDArray pooled, ret;
        // n = num examples, c = num channels or depth
        int n, c, kh, kw, outWidth, outHeight;
        if(training && conf.getDropOut() > 0) {
            this.dropoutMask = Dropout.applyDropout(input,conf.getDropOut(),dropoutMask);
        }

        switch(conf.getPoolingType()) {
            case AVG:
                pooled = Convolution.im2col(input,conf.getKernelSize(),conf.getStride(),conf.getPadding());
                n = pooled.size(0);
                c = pooled.size(1);
                kh = pooled.size(2);
                kw = pooled.size(3);
                outWidth = pooled.size(4);
                outHeight = pooled.size(5);
                ret = pooled.reshape(n, c, kh * kw, outHeight, outWidth);
                return ret.mean(2);
            case MAX:
                pooled = Convolution.im2col(input,conf.getKernelSize(),conf.getStride(),conf.getPadding());
                n = pooled.size(0);
                c = pooled.size(1);
                kh = pooled.size(2);
                kw = pooled.size(3);
                outWidth = pooled.size(4);
                outHeight = pooled.size(5);
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
        throw new UnsupportedOperationException();
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
    public ConvexOptimizer getOptimizer() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void fit() {}

    @Override
    public void fit(INDArray input) {}

    @Override
    public void computeGradientAndScore() {}

    @Override
    public double score() {
        return 0;
    }

    @Override
    public void accumulateScore(double accum) { throw new UnsupportedOperationException(); }

    @Override
    public Gradient gradient() {
        if (gradient== null)
            throw new NullPointerException("Gradient has not been set");
        return gradient;
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray params() {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray getParam(String param) {
        throw new UnsupportedOperationException();
    }
    @Override
    public void setParams(INDArray params) {
        throw new UnsupportedOperationException();
    }


}
