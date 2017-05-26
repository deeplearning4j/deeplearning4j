package org.deeplearning4j.zoo.model;

import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.GaussianDistribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 * AlexNet
 *
 * Dl4j's AlexNet model interpretation based on the original paper ImageNet Classification with Deep Convolutional Neural Networks
 * and the imagenetExample code referenced.
 *
 * References:
 * http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
 * https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/train_val.prototxt
 *
 * Model is built in dl4j based on available functionality and notes indicate where there are gaps waiting for enhancements.
 *
 * Bias initialization in the paper is 1 in certain layers but 0.1 in the imagenetExample code
 * Weight distribution uses 0.1 std for all layers in the paper but 0.005 in the dense layers in the imagenetExample code
 *
 */
@NoArgsConstructor
public class AlexNet extends ZooModel {

    private int[] inputShape = new int[] {3, 224, 224};
    private int numLabels = 1000;
    private long seed = 42;
    private int iterations = 1;
    private WorkspaceMode workspaceMode;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode;

    public AlexNet(int numLabels, long seed, int iterations) {
        this(numLabels, seed, iterations, WorkspaceMode.SEPARATE);
    }

    public AlexNet(int numLabels, long seed, int iterations, WorkspaceMode workspaceMode) {
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
        return ZooType.ALEXNET;
    }

    @Override
    public Class<? extends Model> modelType() {
        return MultiLayerNetwork.class;
    }

    public MultiLayerConfiguration conf() {
        double nonZeroBias = 1;
        double dropOut = 0.5;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
                        .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0.0, 0.01))
                        .activation(Activation.RELU).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .updater(Updater.NESTEROVS).learningRate(1e-2).biasLearningRate(1e-2 * 2).regularization(true)
                        .convolutionMode(ConvolutionMode.Same)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer) // normalize to prevent vanishing or exploding gradients
                        .trainingWorkspaceMode(WorkspaceMode.SINGLE).inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                        .dropOut(0.5).l2(5 * 1e-4).miniBatch(false)
                        .list().layer(0,
                                        new ConvolutionLayer.Builder(new int[] {11, 11}, new int[] {4, 4},
                                                        new int[] {2, 2}).name("cnn1")
                                                                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                                                                        .convolutionMode(ConvolutionMode.Truncate)
                                                                        .nIn(inputShape[0]).nOut(64).build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3},
                                        new int[] {2, 2}, new int[] {1, 1}).convolutionMode(ConvolutionMode.Truncate)
                                                        .name("maxpool1").build())
                        .layer(2, new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {2, 2}, new int[] {2, 2}) // TODO: fix input and put stride back to 1,1
                                        .convolutionMode(ConvolutionMode.Truncate).name("cnn2")
                                        .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).nOut(192)
                                        .biasInit(nonZeroBias).build())
                        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3},
                                        new int[] {2, 2}).name("maxpool2").build())
                        .layer(4, new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {1, 1}, new int[] {1, 1})
                                        .name("cnn3").cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).nOut(384)
                                        .build())
                        .layer(5, new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {1, 1}, new int[] {1, 1})
                                        .name("cnn4").cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).nOut(256)
                                        .biasInit(nonZeroBias).build())
                        .layer(6, new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {1, 1}, new int[] {1, 1})
                                        .name("cnn5").cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST).nOut(256)
                                        .biasInit(nonZeroBias).build())
                        .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {3, 3},
                                        new int[] {7, 7}) // TODO: fix input and put stride back to 2,2
                                                        .name("maxpool3").build())
                        .layer(8, new DenseLayer.Builder().name("ffn1").nIn(256).nOut(4096)
                                        .dist(new GaussianDistribution(0, 0.005)).biasInit(nonZeroBias).dropOut(dropOut)
                                        .build())
                        .layer(9, new DenseLayer.Builder().name("ffn2").nOut(4096)
                                        .dist(new GaussianDistribution(0, 0.005)).biasInit(nonZeroBias).dropOut(dropOut)
                                        .build())
                        .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .name("output").nOut(numLabels).activation(Activation.SOFTMAX).build())
                        .backprop(true).pretrain(false)
                        .setInputType(InputType.convolutionalFlat(inputShape[2], inputShape[1], inputShape[0])).build();

        return conf;
    }

    @Override
    public MultiLayerNetwork init() {
        MultiLayerConfiguration conf = conf();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
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
