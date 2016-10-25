package org.deeplearning4j.nn.conf.module;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;

/**
 * Inception is based on GoogleLeNet configuration of convolutional layers for optimization of resources and learning.
 *
 * This module is based on the Inception GraphBuilderModule built for Torch and a Scala implementation of GoogleLeNet.
 * https://github.com/Element-Research/dpnn/blob/master/Inception.lua
 * https://gist.github.com/antikantian/f77e91f924614348ea8f64731437930d
 *
 * @author Justin Long (crockpotveggies)
 */
public class Inception extends GraphBuilderModule {
    protected String moduleName = "inception";


    @Override public String getModuleName() {
        return this.moduleName;
    }


    public ConvolutionLayer conv1x1(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[] {1,1}, new int[] {1,1}, new int[] {0,0}).nIn(in).nOut(out).biasInit(bias).build();
    }

    public ConvolutionLayer c3x3reduce(int in, int out, double bias) {
        return conv1x1(in, out, bias);
    }

    public ConvolutionLayer c5x5reduce(int in, int out, double bias) {
        return conv1x1(in, out, bias);
    }

    public ConvolutionLayer conv3x3(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).nIn(in).nOut(out).biasInit(bias).build();
    }

    public ConvolutionLayer conv5x5(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{5,5}, new int[] {1,1}, new int[] {2,2}).nIn(in).nOut(out).biasInit(bias).build();
    }

    public ConvolutionLayer conv7x7(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{7,7}, new int[]{2,2}, new int[]{3,3}).nIn(in).nOut(out).biasInit(bias).build();
    }

    public SubsamplingLayer avgPool7x7(int stride) {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{7,7}, new int[]{1,1}).build();
    }

    public SubsamplingLayer maxPool3x3(int stride) {
        return new SubsamplingLayer.Builder(new int[]{3,3}, new int[]{stride,stride}, new int[]{1,1}).build();
    }

    public DenseLayer fullyConnected(int in, int out, double dropOut) {
        return new DenseLayer.Builder().nIn(in).nOut(out).dropOut(dropOut).build();
    }

    /**
     * Appends inception layer configurations a GraphBuilder object, based on the concept of
     * Inception via the GoogleLeNet paper: https://arxiv.org/abs/1409.4842
     *
     * @param graph
     * @param layerName
     * @param inputSize
     * @param config
     * @param inputLayer
     *
     * @return An instance of GraphBuilder with appended inception layers.
     */
    @Override public ComputationGraphConfiguration.GraphBuilder updateBuilder(ComputationGraphConfiguration.GraphBuilder graph, String layerName, int inputSize, int[][] config, String inputLayer) {
        graph
            .addLayer(getModuleName() + "-" + layerName + "-cnn1", conv1x1(inputSize, config[0][0], 0.2), inputLayer)
            .addLayer(getModuleName() + "-" + layerName + "-cnn2", c3x3reduce(inputSize, config[1][0], 0.2), inputLayer)
            .addLayer(getModuleName() + "-" + layerName + "-cnn3", c5x5reduce(inputSize, config[2][0], 0.2), inputLayer)
            .addLayer(getModuleName() + "-" + layerName + "-max1", maxPool3x3(1), inputLayer)
            .addLayer(getModuleName() + "-" + layerName + "-cnn4", conv3x3(config[1][0], config[1][1], 0.2), layerName + "-cnn2")
            .addLayer(getModuleName() + "-" + layerName + "-cnn5", conv5x5(config[2][0], config[2][1], 0.2), layerName + "-cnn3")
            .addLayer(getModuleName() + "-" + layerName + "-cnn6", conv1x1(inputSize, config[3][0], 0.2), layerName + "-max1")
            .addVertex(getModuleName() + "-" + layerName + "-depthconcat1", new MergeVertex(), layerName + "-cnn1", layerName + "-cnn4", layerName + "-cnn5", layerName + "-cnn6");
        return graph;
    }

}
