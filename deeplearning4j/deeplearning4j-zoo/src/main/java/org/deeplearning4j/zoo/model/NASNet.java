package org.deeplearning4j.zoo.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.distribution.TruncatedNormalDistribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import static org.deeplearning4j.zoo.model.helper.NASNetHelper.normalA;
import static org.deeplearning4j.zoo.model.helper.NASNetHelper.reductionA;

/**
 * U-Net
 *
 * Implementation of NASNet-A in Deeplearning4j. NASNet refers to Neural Architecture Search Network, a family of models
 * that were designed automatically by learning the model architectures directly on the dataset of interest.
 *
 * <p>This implementation uses 1056 penultimate filters and an input shape of (3, 224, 224). You can change this.</p>
 *
 * <p>Paper: https://arxiv.org/abs/1707.07012</p>
 * <p>ImageNet weights for this model are available and have been converted from https://keras.io/applications/.</p>
 *
 * @note If using the IMAGENETLARGE weights, the input shape is (3, 331, 331).
 * @author Justin Long (crockpotveggies)
 *
 */
@AllArgsConstructor
@Builder
public class NASNet extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 224, 224};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private WeightInit weightInit = WeightInit.RELU;
    @Builder.Default private Distribution weightDistribution = new TruncatedNormalDistribution(0.0, 0.5); // if WeightInit.DISTRIBUTION
    @Builder.Default private IUpdater updater = new AdaDelta();
    @Builder.Default private CacheMode cacheMode = CacheMode.DEVICE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    // NASNet specific
    @Builder.Default private int numBlocks = 6;
    @Builder.Default private int penultimateFilters = 1056;
    @Builder.Default private int stemFilters = 96;
    @Builder.Default private int filterMultiplier = 2;
    @Builder.Default private boolean skipReduction = true;

    private NASNet() {}

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return DL4JResources.getURLString("models/nasnetmobile_dl4j_inference.v1.zip");
        else if (pretrainedType == PretrainedType.IMAGENETLARGE)
            return DL4JResources.getURLString("models/nasnetlarge_dl4j_inference.v1.zip");
        else
            return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return 3082463801L;
        else if (pretrainedType == PretrainedType.IMAGENETLARGE)
            return 321395591L;
        else
            return 0L;
    }

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    @Override
    public ComputationGraph init() {
        ComputationGraphConfiguration.GraphBuilder graph = graphBuilder();

        graph.addInputs("input").setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]));

        ComputationGraphConfiguration conf = graph.build();
        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }

    public ComputationGraphConfiguration.GraphBuilder graphBuilder() {

        if(penultimateFilters % 24 != 0) {
            throw new IllegalArgumentException("For NASNet-A models penultimate filters must be divisible by 24. Current value is "+penultimateFilters);
        }
        int filters = (int) Math.floor(penultimateFilters / 24);

        ComputationGraphConfiguration.GraphBuilder graph = new NeuralNetConfiguration.Builder().seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(updater)
                .weightInit(weightInit)
                .dist(weightDistribution)
                .l2(5e-5)
                .miniBatch(true)
                .cacheMode(cacheMode)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .cudnnAlgoMode(cudnnAlgoMode)
                .convolutionMode(ConvolutionMode.Truncate)
                .graphBuilder();

        if(!skipReduction) {
            graph.addLayer("stem_conv1", new ConvolutionLayer.Builder(3, 3).stride(2, 2).nOut(stemFilters).hasBias(false)
                    .cudnnAlgoMode(cudnnAlgoMode).build(), "input");
        } else {
            graph.addLayer("stem_conv1", new ConvolutionLayer.Builder(3, 3).stride(1, 1).nOut(stemFilters).hasBias(false)
                    .cudnnAlgoMode(cudnnAlgoMode).build(), "input");
        }

        graph.addLayer("stem_bn1", new BatchNormalization.Builder().eps(1e-3).gamma(0.9997).build(), "stem_conv1");

        String inputX = "stem_bn1";
        String inputP = null;
        if(!skipReduction) {
            Pair<String, String> stem1 = reductionA(graph, (int) Math.floor(stemFilters / Math.pow(filterMultiplier,2)), "stem1", "stem_conv1", inputP);
            Pair<String, String> stem2 = reductionA(graph, (int) Math.floor(stemFilters / (filterMultiplier)), "stem2", stem1.getFirst(), stem1.getSecond());
            inputX = stem2.getFirst();
            inputP = stem2.getSecond();
        }

        for(int i = 0; i < numBlocks; i++){
            Pair<String, String> block = normalA(graph, filters, String.valueOf(i), inputX, inputP);
            inputX = block.getFirst();
            inputP = block.getSecond();
        }

        String inputP0;
        Pair<String, String> reduce = reductionA(graph, filters * filterMultiplier, "reduce"+numBlocks, inputX, inputP);
        inputX = reduce.getFirst();
        inputP0 = reduce.getSecond();

        if(!skipReduction) inputP = inputP0;

        for(int i = 0; i < numBlocks; i++){
            Pair<String, String> block = normalA(graph, filters * filterMultiplier, String.valueOf(i+numBlocks+1), inputX, inputP);
            inputX = block.getFirst();
            inputP = block.getSecond();
        }

        reduce = reductionA(graph, filters * (int)Math.pow(filterMultiplier, 2), "reduce"+(2*numBlocks), inputX, inputP);
        inputX = reduce.getFirst();
        inputP0 = reduce.getSecond();

        if(!skipReduction) inputP = inputP0;

        for(int i = 0; i < numBlocks; i++){
            Pair<String, String> block = normalA(graph, filters * (int) Math.pow(filterMultiplier, 2), String.valueOf(i+(2*numBlocks)+1), inputX, inputP);
            inputX = block.getFirst();
            inputP = block.getSecond();
        }

        // output
        graph
                .addLayer("act", new ActivationLayer(Activation.RELU), inputX)
                .addLayer("avg_pool", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "act")
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .activation(Activation.SOFTMAX).build(), "avg_pool")

                .setOutputs("output")
                .backprop(true)
                .pretrain(false);

        return graph;
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
