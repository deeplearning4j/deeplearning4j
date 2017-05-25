package org.deeplearning4j.zoo.model;

import lombok.Setter;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.ZooType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by kepricon on 17. 3. 30.
 * LeNet
 * Reference: http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
 */
public class LeNet extends ZooModel {

    private int[] inputShape = new int[] {3, 224, 224};
    private int numLabels;
    private long seed;
    private int iterations;
    private WorkspaceMode workspaceMode;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode;

    public LeNet(int numLabels, long seed, int iterations) {
        this(numLabels, seed, iterations, WorkspaceMode.SEPARATE);
    }

    public LeNet(int numLabels, long seed, int iterations, WorkspaceMode workspaceMode) {
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
        this.workspaceMode = workspaceMode;
        this.cudnnAlgoMode = workspaceMode == WorkspaceMode.SINGLE ? ConvolutionLayer.AlgoMode.PREFER_FASTEST
                        : ConvolutionLayer.AlgoMode.NO_WORKSPACE;
    }

    @Override
    public String pretrainedImageNetUrl() {
        return null;
    }

    @Override
    public String pretrainedMnistUrl() {
        return "http://blob.deeplearning4j.org/models/lenet_dl4j_mnist_inference.zip";
    }

    @Override
    public ZooType zooType() {
        return ZooType.LENET;
    }

    @Override
    public Class<? extends Model> modelType() {
        return MultiLayerNetwork.class;
    }

    public MultiLayerConfiguration conf() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .trainingWorkspaceMode(workspaceMode)
                        .inferenceWorkspaceMode(workspaceMode)
                        .seed(seed)
                        .iterations(iterations)
                        .activation(Activation.IDENTITY)
                        .weightInit(WeightInit.RELU)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new AdaDelta())
                        .regularization(false)
                        .convolutionMode(ConvolutionMode.Same)
                        .list()
                        // block 1
                        .layer(0, new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {1, 1}).name("cnn1")
                                        .nIn(inputShape[0]).nOut(20).activation(Activation.RELU).build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2},
                                        new int[] {2, 2}).name("maxpool1").build())
                        // block 2
                        .layer(2, new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {1, 1}).name("cnn2").nOut(50)
                                        .activation(Activation.RELU).build())
                        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2},
                                        new int[] {2, 2}).name("maxpool2").build())
                        // fully connected
                        .layer(4, new DenseLayer.Builder().name("ffn1").activation(Activation.RELU).nOut(500).build())
                        // output
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                        .name("output").nOut(numLabels).activation(Activation.SOFTMAX) // radial basis function required
                                        .build())
                        .setInputType(InputType.convolutionalFlat(inputShape[2], inputShape[1], inputShape[0]))
                        .backprop(true).pretrain(false).build();

        return conf;
    }

    @Override
    public Model init() {
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
