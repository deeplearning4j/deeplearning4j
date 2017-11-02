package org.deeplearning4j.zoo.model;

import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

/**
 * Tiny YOLO
 *  Reference: https://arxiv.org/pdf/1612.08242.pdf
 *
 * <p>ImageNet+VOC weights for this model are available and have been converted from https://pjreddie.com/darknet/yolo/
 * using https://github.com/allanzelener/YAD2K and the following code.</p>
 *
 * <pre>{@code
 *     String filename = "tiny-yolo-voc.h5";
 *     ComputationGraph graph = KerasModelImport.importKerasModelAndWeights(filename, false);
 *     INDArray priors = Nd4j.create(priorBoxes);
 *
 *     FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
 *             .seed(seed)
 *             .iterations(iterations)
 *             .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
 *             .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
 *             .gradientNormalizationThreshold(1.0)
 *             .updater(new Adam.Builder().learningRate(1e-3).build())
 *             .l2(0.00001)
 *             .activation(Activation.IDENTITY)
 *             .trainingWorkspaceMode(workspaceMode)
 *             .inferenceWorkspaceMode(workspaceMode)
 *             .build();
 *
 *     ComputationGraph model = new TransferLearning.GraphBuilder(graph)
 *             .fineTuneConfiguration(fineTuneConf)
 *             .addLayer("outputs",
 *                     new Yolo2OutputLayer.Builder()
 *                             .boundingBoxPriors(priors)
 *                             .build(),
 *                     "conv2d_9")
 *             .setOutputs("outputs")
 *             .build();
 *
 *     System.out.println(model.summary(InputType.convolutional(416, 416, 3)));
 *
 *     ModelSerializer.writeModel(model, "tiny-yolo-voc_dl4j_inference.v1.zip", false);
 *}</pre>
 *
 * The channels of the 416x416 input images need to be in RGB order (not BGR), with values normalized within [0, 1].
 *
 * @author saudet
 */
@NoArgsConstructor
public class TinyYOLO extends ZooModel {

    public static int nBoxes = 5;
    public static double[][] priorBoxes = {{1.08, 1.19}, {3.42, 4.41}, {6.63, 11.38}, {9.42, 5.11}, {16.62, 10.52}};

    private int[] inputShape = {3, 416, 416};
    private int numLabels;
    private long seed;
    private int iterations;
    private WorkspaceMode workspaceMode;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode;

    public TinyYOLO(int numLabels, long seed, int iterations) {
        this(numLabels, seed, iterations, WorkspaceMode.SEPARATE);
    }

    public TinyYOLO(int numLabels, long seed, int iterations, WorkspaceMode workspaceMode) {
        this.numLabels = numLabels;
        this.seed = seed;
        this.iterations = iterations;
        this.workspaceMode = workspaceMode;
        this.cudnnAlgoMode = workspaceMode == WorkspaceMode.SINGLE ? ConvolutionLayer.AlgoMode.PREFER_FASTEST
                        : ConvolutionLayer.AlgoMode.NO_WORKSPACE;
    }

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return "http://blob.deeplearning4j.org/models/tiny-yolo-voc_dl4j_inference.v1.zip";
        else
            return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return 2004171617L;
        else
            return 0L;
    }

    @Override
    public ZooType zooType() {
        return ZooType.TINYYOLO;
    }

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    private GraphBuilder addLayers(GraphBuilder graphBuilder, int layerNumber, int filterSize, int nIn, int nOut, int poolSize, int poolStride) {
        String input = "maxpooling2d_" + (layerNumber - 1);
        if (!graphBuilder.getVertices().containsKey(input)) {
            input = "activation_" + (layerNumber - 1);
        }
        if (!graphBuilder.getVertices().containsKey(input)) {
            input = "input";
        }

        graphBuilder
                .addLayer("convolution2d_" + layerNumber,
                        new ConvolutionLayer.Builder(filterSize,filterSize)
                                .nIn(nIn)
                                .nOut(nOut)
                                .weightInit(WeightInit.XAVIER)
                                .convolutionMode(ConvolutionMode.Same)
                                .hasBias(false)
                                .stride(1,1)
                                .activation(Activation.IDENTITY)
                                .cudnnAlgoMode(cudnnAlgoMode)
                                .build(),
                        input)
                .addLayer("batchnormalization_" + layerNumber,
                        new BatchNormalization.Builder()
                                .nIn(nOut).nOut(nOut)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "convolution2d_" + layerNumber)
                .addLayer("activation_" + layerNumber,
                        new ActivationLayer.Builder()
                                .activation(new ActivationLReLU(0.1))
                                .build(),
                        "batchnormalization_" + layerNumber);
        if (poolSize > 0) {
            graphBuilder
                    .addLayer("maxpooling2d_" + layerNumber,
                            new SubsamplingLayer.Builder()
                                    .kernelSize(poolSize, poolSize)
                                    .stride(poolStride, poolStride)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .build(),
                            "activation_" + layerNumber);
        }

        return graphBuilder;
    }

    public ComputationGraphConfiguration conf() {
        INDArray priors = Nd4j.create(priorBoxes);

        GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Adam.Builder().learningRate(1e-3).build())
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]));

        addLayers(graphBuilder, 1, 3, inputShape[0], 16, 2, 2);

        addLayers(graphBuilder, 2, 3, 16, 32, 2, 2);

        addLayers(graphBuilder, 3, 3, 32, 64, 2, 2);

        addLayers(graphBuilder, 4, 3, 64, 128, 2, 2);

        addLayers(graphBuilder, 5, 3, 128, 256, 2, 2);

        addLayers(graphBuilder, 6, 3, 256, 512, 2, 1);

        addLayers(graphBuilder, 7, 3, 512, 1024, 0, 0);
        addLayers(graphBuilder, 8, 3, 1024, 1024, 0, 0);

        int layerNumber = 9;
        graphBuilder
                .addLayer("convolution2d_" + layerNumber,
                        new ConvolutionLayer.Builder(1,1)
                                .nIn(1024)
                                .nOut(nBoxes * (5 + numLabels))
                                .weightInit(WeightInit.XAVIER)
                                .stride(1,1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.RELU)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "activation_" + (layerNumber - 1))
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .boundingBoxPriors(priors)
                                .build(),
                        "convolution2d_" + layerNumber)
                .setOutputs("outputs").backprop(true).pretrain(false);

        return graphBuilder.build();
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

    /** Returns {@code inputShape[1] / 32}, where {@code inputShape[1]} should be a multiple of 32. */
    public int getGridWidth() {
        return inputShape[1] / 32;
    }

    /** Returns {@code inputShape[2] / 32}, where {@code inputShape[2]} should be a multiple of 32. */
    public int getGridHeight() {
        return inputShape[2] / 32;
    }
}
