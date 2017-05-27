package org.deeplearning4j.zoo.model;

import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * GoogleLeNet
 *  Reference: http://arxiv.org/pdf/1409.4842v1.pdf
 *
 *  Revised and consolidated version by @kedardoshi
 *
 * <p>ImageNet weights for this model are available and have been converted from https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14.</p>
 *
 * @author kedardoshi
 * @author Justin Long (crockpotveggies)
 */
@NoArgsConstructor
public class GoogLeNet extends ZooModel {

    private int[] inputShape = new int[] {3, 224, 224};
    private int numLabels;
    private long seed;
    private int iterations;
    private WorkspaceMode workspaceMode;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode;

    public GoogLeNet(int numLabels, long seed, int iterations) {
        this(numLabels, seed, iterations, WorkspaceMode.SEPARATE);
    }

    public GoogLeNet(int numLabels, long seed, int iterations, WorkspaceMode workspaceMode) {
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
            return "http://blob.deeplearning4j.org/models/googlenet_dl4j_inference.zip";
        else
            return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        if(pretrainedType==PretrainedType.IMAGENET)
            return 3337733202L;
        else
            return 0L;
    }

    @Override
    public ZooType zooType() {
        return ZooType.GOOGLENET;
    }

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    private ConvolutionLayer conv1x1(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[] {1, 1}, new int[] {1, 1}, new int[] {0, 0}).nIn(in).nOut(out)
                        .biasInit(bias).build();
    }

    private ConvolutionLayer c3x3reduce(int in, int out, double bias) {
        return conv1x1(in, out, bias);
    }

    private ConvolutionLayer c5x5reduce(int in, int out, double bias) {
        return conv1x1(in, out, bias);
    }

    private ConvolutionLayer conv3x3(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[] {3, 3}, new int[] {1, 1}, new int[] {1, 1}).nIn(in).nOut(out)
                        .biasInit(bias).build();
    }

    private ConvolutionLayer conv5x5(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[] {5, 5}, new int[] {1, 1}, new int[] {2, 2}).nIn(in).nOut(out)
                        .biasInit(bias).build();
    }

    private ConvolutionLayer conv7x7(int in, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[] {7, 7}, new int[] {2, 2}, new int[] {3, 3}).nIn(in).nOut(out)
                        .biasInit(bias).build();
    }

    private SubsamplingLayer avgPool7x7(int stride) {
        return new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG, new int[] {7, 7}, new int[] {1, 1})
                        .build();
    }

    private SubsamplingLayer maxPool3x3(int stride) {
        return new SubsamplingLayer.Builder(new int[] {3, 3}, new int[] {stride, stride}, new int[] {1, 1}).build();
    }

    private DenseLayer fullyConnected(int in, int out, double dropOut) {
        return new DenseLayer.Builder().nIn(in).nOut(out).dropOut(dropOut).build();
    }

    private GraphBuilder inception(GraphBuilder graph, String name, int inputSize, int[][] config, String inputLayer) {
        graph.addLayer(name + "-cnn1", conv1x1(inputSize, config[0][0], 0.2), inputLayer)
                        .addLayer(name + "-cnn2", c3x3reduce(inputSize, config[1][0], 0.2), inputLayer)
                        .addLayer(name + "-cnn3", c5x5reduce(inputSize, config[2][0], 0.2), inputLayer)
                        .addLayer(name + "-max1", maxPool3x3(1), inputLayer)
                        .addLayer(name + "-cnn4", conv3x3(config[1][0], config[1][1], 0.2), name + "-cnn2")
                        .addLayer(name + "-cnn5", conv5x5(config[2][0], config[2][1], 0.2), name + "-cnn3")
                        .addLayer(name + "-cnn6", conv1x1(inputSize, config[3][0], 0.2), name + "-max1")
                        .addVertex(name + "-depthconcat1", new MergeVertex(), name + "-cnn1", name + "-cnn4",
                                        name + "-cnn5", name + "-cnn6");
        return graph;
    }

    public ComputationGraphConfiguration conf() {
        GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
                        .activation(Activation.RELU).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .learningRate(1e-2).biasLearningRate(2 * 1e-2).learningRateDecayPolicy(LearningRatePolicy.Step)
                        .lrPolicyDecayRate(0.96).lrPolicySteps(320000).updater(new Nesterovs(1e-2, 0.9))
                        .weightInit(WeightInit.XAVIER).regularization(true).l2(2e-4).graphBuilder();

        graph.addInputs("input").addLayer("cnn1", conv7x7(inputShape[0], 64, 0.2), "input")
                        .addLayer("max1",
                                        new SubsamplingLayer.Builder(new int[] {3, 3}, new int[] {2, 2},
                                                        new int[] {0, 0}).build(),
                                        "cnn1")
                        .addLayer("lrn1", new LocalResponseNormalization.Builder(5, 1e-4, 0.75).build(), "max1")
                        .addLayer("cnn2", conv1x1(64, 64, 0.2), "lrn1").addLayer("cnn3", conv3x3(64, 192, 0.2), "cnn2")
                        .addLayer("lrn2", new LocalResponseNormalization.Builder(5, 1e-4, 0.75).build(), "cnn3")
                        .addLayer("max2", new SubsamplingLayer.Builder(new int[] {3, 3}, new int[] {2, 2},
                                        new int[] {0, 0}).build(), "lrn2");

        inception(graph, "3a", 192, new int[][] {{64}, {96, 128}, {16, 32}, {32}}, "max2");
        inception(graph, "3b", 256, new int[][] {{128}, {128, 192}, {32, 96}, {64}}, "3a-depthconcat1");
        graph.addLayer("max3",
                        new SubsamplingLayer.Builder(new int[] {3, 3}, new int[] {2, 2}, new int[] {0, 0}).build(),
                        "3b-depthconcat1");
        inception(graph, "4a", 480, new int[][] {{192}, {96, 208}, {16, 48}, {64}}, "3b-depthconcat1");
        inception(graph, "4b", 512, new int[][] {{160}, {112, 224}, {24, 64}, {64}}, "4a-depthconcat1");
        inception(graph, "4c", 512, new int[][] {{128}, {128, 256}, {24, 64}, {64}}, "4b-depthconcat1");
        inception(graph, "4d", 512, new int[][] {{112}, {144, 288}, {32, 64}, {64}}, "4c-depthconcat1");
        inception(graph, "4e", 528, new int[][] {{256}, {160, 320}, {32, 128}, {128}}, "4d-depthconcat1");
        graph.addLayer("max4",
                        new SubsamplingLayer.Builder(new int[] {3, 3}, new int[] {2, 2}, new int[] {0, 0}).build(),
                        "4e-depthconcat1");
        inception(graph, "5a", 832, new int[][] {{256}, {160, 320}, {32, 128}, {128}}, "max4");
        inception(graph, "5b", 832, new int[][] {{384}, {192, 384}, {48, 128}, {128}}, "5a-depthconcat1");
        graph.addLayer("avg3", avgPool7x7(1), "5b-depthconcat1") // output: 1x1x1024
                        .addLayer("fc1", fullyConnected(1024, 1024, 0.4), "avg3") // output: 1x1x1024
                        .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .nIn(1024).nOut(numLabels).activation(Activation.SOFTMAX).build(), "fc1")
                        .setOutputs("output").backprop(true).pretrain(false);

        return graph.build();
    }

    @Override
    public ComputationGraph init() {
        ComputationGraph model = new ComputationGraph(conf());
        model.init();

        return model;
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
