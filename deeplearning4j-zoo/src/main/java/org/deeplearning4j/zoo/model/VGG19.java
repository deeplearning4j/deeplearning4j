package org.deeplearning4j.zoo.model;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * VGG-19, from Very Deep Convolutional Networks for Large-Scale Image Recognition
 * https://arxiv.org/abs/1409.1556)
 *
 * <p>ImageNet weights for this model are available and have been converted from https://github.com/fchollet/keras/tree/1.1.2/keras/applications.</p>
 *
 * @author Justin Long (crockpotveggies)
 */
public class VGG19 extends ZooModel {

    private int[] inputShape = new int[] {3, 224, 224};
    private int numLabels;
    private long seed;
    private int iterations;
    private WorkspaceMode workspaceMode;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode;

    public VGG19(int numLabels, long seed, int iterations) {
        this(numLabels, seed, iterations, WorkspaceMode.SEPARATE);
    }

    public VGG19(int numLabels, long seed, int iterations, WorkspaceMode workspaceMode) {
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
        this.workspaceMode = workspaceMode;
        this.cudnnAlgoMode = workspaceMode == WorkspaceMode.SINGLE ? ConvolutionLayer.AlgoMode.PREFER_FASTEST
                        : ConvolutionLayer.AlgoMode.NO_WORKSPACE;
    }

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        if(pretrainedType==PretrainedType.IMAGENET)
            return "http://blob.deeplearning4j.org/models/vgg19_dl4j_inference.zip";
        else
            return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        if(pretrainedType==PretrainedType.IMAGENET)
            return 2782932419L;
        else
            return 0L;
    }

    @Override
    public ZooType zooType() {
        return ZooType.VGG16;
    }

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    public MultiLayerConfiguration conf() {
        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder()
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .updater(Updater.NESTEROVS).activation(Activation.RELU)
                                        .trainingWorkspaceMode(workspaceMode).inferenceWorkspaceMode(workspaceMode)
                                        .list()
                                        // block 1
                                        .layer(0, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nIn(inputShape[0]).nOut(64)
                                                        .cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(1, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(64).cudnnAlgoMode(
                                                                        cudnnAlgoMode)
                                                        .build())
                                        .layer(2, new SubsamplingLayer.Builder()
                                                        .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                                        .stride(2, 2).build())
                                        // block 2
                                        .layer(3, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(4, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(128).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(5, new SubsamplingLayer.Builder()
                                                        .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                                        .stride(2, 2).build())
                                        // block 3
                                        .layer(6, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(7, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(8, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(9, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(256).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(10, new SubsamplingLayer.Builder()
                                                        .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                                        .stride(2, 2).build())
                                        // block 4
                                        .layer(11, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(12, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(13, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(14, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(15, new SubsamplingLayer.Builder()
                                                        .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                                        .stride(2, 2).build())
                                        // block 5
                                        .layer(16, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(17, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(18, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(19, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(20, new SubsamplingLayer.Builder()
                                                        .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                                        .stride(2, 2).build())
                                        .layer(21, new DenseLayer.Builder().nOut(4096).build())
                                        .layer(22, new OutputLayer.Builder(
                                                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output")
                                                                        .nOut(numLabels).activation(Activation.SOFTMAX) // radial basis function required
                                                                        .build())
                                        .backprop(true).pretrain(false).setInputType(InputType
                                                        .convolutionalFlat(inputShape[2], inputShape[1], inputShape[0]))
                                        .build();

        return conf;
    }

    @Override
    public MultiLayerNetwork init() {
        MultiLayerNetwork network = new MultiLayerNetwork(conf());
        network.init();
        return network;
    }

    @Override
    public ModelMetaData metaData() {
        return new ModelMetaData(new int[][] {inputShape}, 1, ZooType.CNN);
    }

    @Override
    public void setInputShape(int[][] inputShape) {
        this.inputShape = inputShape[0];
    }

}
