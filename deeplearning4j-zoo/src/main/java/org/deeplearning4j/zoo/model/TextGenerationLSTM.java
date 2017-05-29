package org.deeplearning4j.zoo.model;

import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * LSTM designed for text generation. Can be trained on a corpus of text. For this model, numLabels is
 * used to input {@code totalUniqueCharacters} for the LSTM input layer.
 *
 * Architecture follows this implementation: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
 *
 * <p>Walt Whitman weights are available for generating text from his works, adapted from https://github.com/craigomac/InfiniteMonkeys.</p>
 *
 * @author Justin Long (crockpotveggies)
 */
@NoArgsConstructor
public class TextGenerationLSTM extends ZooModel {

    private int maxLength = 40;
    private int totalUniqueCharacters = 47;
    private int[] inputShape = new int[] {maxLength, totalUniqueCharacters};
    private long seed;
    private int iterations;
    private WorkspaceMode workspaceMode;
    private ConvolutionLayer.AlgoMode cudnnAlgoMode;

    public TextGenerationLSTM(int numLabels, long seed, int iterations) {
        this(numLabels, seed, iterations, WorkspaceMode.SEPARATE);
    }

    public TextGenerationLSTM(int numLabels, long seed, int iterations, WorkspaceMode workspaceMode) {
        this.totalUniqueCharacters = numLabels;
        this.inputShape = new int[] {maxLength, totalUniqueCharacters};
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
        return ZooType.TEXTGENLSTM;
    }

    @Override
    public Class<? extends Model> modelType() {
        return MultiLayerNetwork.class;
    }

    public MultiLayerConfiguration conf() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .learningRate(0.01)
                .seed(12345)
                .regularization(true)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(new RmsProp())
                .list()
                .layer(0, new GravesLSTM.Builder().nIn(inputShape[1]).nOut(256).activation(Activation.TANH).build())
                .layer(1, new GravesLSTM.Builder().nOut(256).activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                        .nOut(totalUniqueCharacters).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(50).tBPTTBackwardLength(50)
                .pretrain(false).backprop(true)
                .build();

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
        return new ModelMetaData(new int[][] {inputShape}, 1, ZooType.RNN);
    }

    @Override
    public void setInputShape(int[][] inputShape) {
        this.inputShape = inputShape[0];
    }
}
