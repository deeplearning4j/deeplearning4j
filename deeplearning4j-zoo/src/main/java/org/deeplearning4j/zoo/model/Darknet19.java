package org.deeplearning4j.zoo.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.deeplearning4j.zoo.model.helper.DarknetHelper.addLayers;

/**
 * Darknet19
 *  Reference: https://arxiv.org/pdf/1612.08242.pdf
 *
 * <p>ImageNet weights for this model are available and have been converted from https://pjreddie.com/darknet/imagenet/
 * using https://github.com/allanzelener/YAD2K .</p>
 *
 * There are 2 pretrained models, one for 224x224 images and one fine-tuned for 448x448 images.
 * Call setInputShape() with either {3, 224, 224} or {3, 448, 448} before initialization.
 * The channels of the input images need to be in RGB order (not BGR), with values normalized within [0, 1].
 * The output labels are as per https://github.com/pjreddie/darknet/blob/master/data/imagenet.shortnames.list .
 *
 * @author saudet
 */
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Darknet19 extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = {3, 224, 224};
    private int numClasses;
    @Builder.Default private WeightInit weightInit = WeightInit.RELU;
    @Builder.Default private IUpdater updater = new Nesterovs(1e-3, 0.9);
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            if (inputShape[1] == 448 && inputShape[2] == 448)
                return "http://blob.deeplearning4j.org/models/darknet19_448_dl4j_inference.v1.zip";
            else
                return "http://blob.deeplearning4j.org/models/darknet19_dl4j_inference.v1.zip";
        else
            return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            if (inputShape[1] == 448 && inputShape[2] == 448)
                return 870575230L;
            else
                return 3952910425L;
        else
            return 0L;
    }

    @Override
    public Class<? extends Model> modelType() {
        return ComputationGraph.class;
    }

    public ComputationGraphConfiguration conf() {
        GraphBuilder graphBuilder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(updater)
                .weightInit(weightInit)
                .l2(0.00001)
                .activation(Activation.IDENTITY)
                .cacheMode(cacheMode)
                .trainingWorkspaceMode(workspaceMode)
                .inferenceWorkspaceMode(workspaceMode)
                .cudnnAlgoMode(cudnnAlgoMode)
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutional(inputShape[2], inputShape[1], inputShape[0]));

        addLayers(graphBuilder, 1, 3, inputShape[0],  32, 2);

        addLayers(graphBuilder, 2, 3, 32, 64, 2);

        addLayers(graphBuilder, 3, 3, 64, 128, 0);
        addLayers(graphBuilder, 4, 1, 128, 64, 0);
        addLayers(graphBuilder, 5, 3, 64, 128, 2);

        addLayers(graphBuilder, 6, 3, 128, 256, 0);
        addLayers(graphBuilder, 7, 1, 256, 128, 0);
        addLayers(graphBuilder, 8, 3, 128, 256, 2);

        addLayers(graphBuilder, 9, 3, 256, 512, 0);
        addLayers(graphBuilder, 10, 1, 512, 256, 0);
        addLayers(graphBuilder, 11, 3, 256, 512, 0);
        addLayers(graphBuilder, 12, 1, 512, 256, 0);
        addLayers(graphBuilder, 13, 3, 256, 512, 2);

        addLayers(graphBuilder, 14, 3, 512, 1024, 0);
        addLayers(graphBuilder, 15, 1, 1024, 512, 0);
        addLayers(graphBuilder, 16, 3, 512, 1024, 0);
        addLayers(graphBuilder, 17, 1, 1024, 512, 0);
        addLayers(graphBuilder, 18, 3, 512, 1024, 0);

        int layerNumber = 19;
        graphBuilder
                .addLayer("convolution2d_" + layerNumber,
                        new ConvolutionLayer.Builder(1,1)
                                .nIn(1024)
                                .nOut(numClasses)
                                .weightInit(WeightInit.XAVIER)
                                .stride(1,1)
                                .convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.RELU)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "activation_" + (layerNumber - 1))
                .addLayer("globalpooling", new GlobalPoolingLayer.Builder(PoolingType.AVG)
                        .build(), "convolution2d_" + layerNumber)
                .addLayer("softmax", new ActivationLayer.Builder()
                        .activation(Activation.SOFTMAX)
                        .build(), "globalpooling")
                .addLayer("loss", new LossLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build(), "softmax")
                .setOutputs("loss").backprop(true).pretrain(false);

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
}
