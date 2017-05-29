package org.deeplearning4j.zoo.model;

import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * A simple convolutional network for generic image classification.
 * Reference: https://github.com/oarriaga/face_classification/
 *
 * @author Justin Long (crockpotveggies)
 */
@NoArgsConstructor
public class SimpleCNN extends ZooModel {

    private int[] inputShape = new int[] {3, 48, 48};
    private int numLabels;
    private long seed;
    private int iterations;
    private WorkspaceMode workspaceMode;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode;

    public SimpleCNN(int numLabels, long seed, int iterations) {
        this(numLabels, seed, iterations, WorkspaceMode.SEPARATE);
    }

    public SimpleCNN(int numLabels, long seed, int iterations, WorkspaceMode workspaceMode) {
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
        this.workspaceMode = workspaceMode;
        this.cudnnAlgoMode = workspaceMode == WorkspaceMode.SINGLE ? ConvolutionLayer.AlgoMode.PREFER_FASTEST
                        : ConvolutionLayer.AlgoMode.NO_WORKSPACE;
    }

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        return 0L;
    }

    @Override
    public ZooType zooType() {
        return ZooType.SIMPLECNN;
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
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(new AdaDelta())
                        .regularization(false)
                        .convolutionMode(ConvolutionMode.Same)
                        .list()
                        // block 1
                        .layer(0, new ConvolutionLayer.Builder(new int[]{7,7}).name("image_array")
                                        .nIn(inputShape[0]).nOut(16).build())
                        .layer(1, new BatchNormalization.Builder().build())
                        .layer(2, new ConvolutionLayer.Builder(new int[]{7,7}).nIn(16).nOut(16).build())
                        .layer(3, new BatchNormalization.Builder().build())
                        .layer(4, new ActivationLayer.Builder().activation(Activation.RELU).build())
                        .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{2,2}).build())
                        .layer(6, new DropoutLayer.Builder(0.5).build())

                        // block 2
                        .layer(7, new ConvolutionLayer.Builder(new int[]{5,5}).nOut(32).build())
                        .layer(8, new BatchNormalization.Builder().build())
                        .layer(9, new ConvolutionLayer.Builder(new int[]{5,5}).nOut(32).build())
                        .layer(10, new BatchNormalization.Builder().build())
                        .layer(11, new ActivationLayer.Builder().activation(Activation.RELU).build())
                        .layer(12, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{2,2}).build())
                        .layer(13, new DropoutLayer.Builder(0.5).build())

                        // block 3
                        .layer(14, new ConvolutionLayer.Builder(new int[]{3,3}).nOut(64).build())
                        .layer(15, new BatchNormalization.Builder().build())
                        .layer(16, new ConvolutionLayer.Builder(new int[]{3,3}).nOut(64).build())
                        .layer(17, new BatchNormalization.Builder().build())
                        .layer(18, new ActivationLayer.Builder().activation(Activation.RELU).build())
                        .layer(19, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{2,2}).build())
                        .layer(20, new DropoutLayer.Builder(0.5).build())

                        // block 4
                        .layer(21, new ConvolutionLayer.Builder(new int[]{3,3}).nOut(128).build())
                        .layer(22, new BatchNormalization.Builder().build())
                        .layer(23, new ConvolutionLayer.Builder(new int[]{3,3}).nOut(128).build())
                        .layer(24, new BatchNormalization.Builder().build())
                        .layer(25, new ActivationLayer.Builder().activation(Activation.RELU).build())
                        .layer(26, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[]{2,2}).build())
                        .layer(27, new DropoutLayer.Builder(0.5).build())


                        // block 5
                        .layer(28, new ConvolutionLayer.Builder(new int[]{3,3}).nOut(256).build())
                        .layer(29, new BatchNormalization.Builder().build())
                        .layer(30, new ConvolutionLayer.Builder(new int[]{3,3}).nOut(numLabels).build())
                        .layer(31, new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
                        .layer(32, new ActivationLayer.Builder().activation(Activation.SOFTMAX).build())

                        .setInputType(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]))
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
