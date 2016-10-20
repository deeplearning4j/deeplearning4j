package org.deeplearning4j.nn.conf.module;

import javafx.util.Pair;
import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;

import java.util.ArrayList;

/**
 * Inception is based on GoogleLeNet configuration of convolutional layers for optimization of resources and learning.
 *
 * This module is based on the Inception Module built for Torch and a Scala implementation of GoogleLeNet.
 * https://github.com/Element-Research/dpnn/blob/master/Inception.lua
 * https://gist.github.com/antikantian/f77e91f924614348ea8f64731437930d
 *
 * Uses n+2 parallel "columns". The original Inception paper uses 2+2.
 *
 * @author Justin Long
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class InceptionModule extends Module {
    protected String moduleName = "inception";
    protected int inputSize = 192;
    protected int[] kernelSize = new int[] {3,5};
    protected int[] kernelStride = new int[] {1,1};
    protected int[] outputSize = new int[] {128, 32};
    protected int[] reduceSize = new int[] {96, 16, 32, 64};
    protected SubsamplingLayer pool = new SubsamplingLayer.Builder(new int[]{3,3}, kernelStride, new int[]{1,1}).build();
    protected boolean batchNorm = true;
    protected ArrayList<Pair<String, Layer>> layers = new ArrayList<>();


    private ConvolutionLayer conv1x1(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{1,1}, new int[]{1,1}).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer c3x3reduce(int in, int out, double bias) {
        return conv1x1(in, out, bias);
    }

    private ConvolutionLayer c5x5reduce(int in, int out, double bias) {
        return conv1x1(in, out, bias);
    }

    private ConvolutionLayer conv3x3(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[]{1,1}, new int[]{1,1}).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, new int[]{1,1}, new int[]{2,2}).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv7x7(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{7,7}, new int[]{2,2}, new int[]{3,3}).nIn(in).nOut(out).biasInit(bias).build();
    }

    private SubsamplingLayer avgPool7x7(int stride) {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{7,7}, new int[]{1,1}).build();
    }

    private SubsamplingLayer maxPool3x3(int stride) {
        return new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{stride,stride}, new int[]{1,1}).build();
    }

    private DenseLayer fullyConnected(int in, int out, double dropOut) {
        return new DenseLayer.Builder().nIn(in).nOut(out).dropOut(dropOut).build();
    }

    private MergeVertex mergeVertex() {
        return new MergeVertex();
    }


    /**
    * ConvolutionLayer
    * nIn in the input layer is the number of channels
    * nOut is the number of filters to be used in the net or in other words the depth
    * The builder specifies the filter/kernel size, the kernelStride and outputSize
    * The pooling layer takes the kernel size
    */
    private InceptionModule(Builder builder) {
        if(builder.kernelSize.length != 2)
            throw new IllegalArgumentException("Kernel size of should be rows x columns (a 2d array)");
        this.kernelSize = builder.kernelSize;
        if(builder.kernelStride.length != 2)
            throw new IllegalArgumentException("Stride should include kernelStride for rows and columns (a 2d array)");
        this.kernelStride = builder.kernelStride;
        if(builder.outputSize.length != 2)
            throw new IllegalArgumentException("Padding should include outputSize for rows and columns (a 2d array)");
        this.outputSize = builder.outputSize;


        for (int i = 0; i < kernelSize.length; i++) {
            this.addLayer("cnn1", conv1x1(this.inputSize, this.outputSize[i], 0.02));
            this.addLayer("cnn2", c3x3reduce(this.inputSize, this.outputSize[i], 0.02));
            this.addLayer("cnn3", c5x5reduce(this.inputSize, this.outputSize[i], 0.02));
            maxPool3x3(1);
            //TODO: need to calculate in/out among other things
//            conv3x3(this.inputSize, this.outputSize[i], 0.02);
//            conv5x5(this.inputSize, this.outputSize[i], 0.02);
//            conv1x1(this.inputSize, this.outputSize[i], 0.02);
        }
    }

    @Override
    public void addLayer(String layerName, Layer layer) {
        layers.add(new Pair(layerName, layer));
    }

    public void addLayer(Pair<String, Layer> layerPair) {
        layers.add(layerPair);
    }

    @Override
    public InceptionModule clone() {
        InceptionModule clone = new InceptionModule(
            this.inputSize, this.kernelSize.clone(), this.kernelStride.clone(), this.outputSize.clone(), this.reduceSize.clone(), this.pool.clone(), this.batchNorm)

        for (Pair<String, Layer> layerPair : this.layers) {
            clone.addLayer(new Pair<>(layerPair.getKey(), layerPair.getValue().clone()));
        }

        return clone;
    }

    @AllArgsConstructor
    public static class Builder extends FeedForwardLayer.Builder<Builder> {
        private int inputSize = 192;
        private int[] kernelSize = new int[] {3,5};
        private int[] kernelStride = new int[] {1,1};
        private int[] outputSize = new int[] {128, 32};
        private int[] reduceSize = new int[] {96, 16, 32, 64};
        private SubsamplingLayer pool = new SubsamplingLayer.Builder(new int[]{3,3}, kernelStride, new int[]{1,1}).build();
        private boolean batchNorm = true;

        public Builder(int inputSize, int[] kernelSize, int[] kernelStride, int[] outputSize, int[] reduceSize, SubsamplingLayer pool, boolean batchNorm) {
            this.inputSize = inputSize;
            this.kernelSize = kernelSize;
            this.kernelStride = kernelStride;
            this.outputSize = outputSize;
            this.reduceSize = reduceSize;
            this.pool = pool;
            this.batchNorm = batchNorm;
        }

        @Override
        @SuppressWarnings("unchecked")
        public InceptionModule build() {
            return new InceptionModule(this);
        }
    }
}
