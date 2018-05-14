package org.deeplearning4j.zoo.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.zoo.ModelMetaData;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.ZooType;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * VGG-16, from Very Deep Convolutional Networks for Large-Scale Image Recognition
 * https://arxiv.org/abs/1409.1556
 *
 * Deep Face Recognition
 * http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf
 *
 * <p>ImageNet weights for this model are available and have been converted from https://github.com/fchollet/keras/tree/1.1.2/keras/applications.</p>
 * <p>CIFAR-10 weights for this model are available and have been converted using "approach 2" from https://github.com/rajatvikramsingh/cifar10-vgg16.</p>
 * <p>VGGFace weights for this model are available and have been converted from https://github.com/rcmalli/keras-vggface.</p>
 *
 * @author Justin Long (crockpotveggies)
 */
@AllArgsConstructor
@Builder
public class VGG16 extends ZooModel {

    @Builder.Default private long seed = 1234;
    @Builder.Default private int[] inputShape = new int[] {3, 224, 224};
    @Builder.Default private int numClasses = 0;
    @Builder.Default private IUpdater updater = new Nesterovs();
    @Builder.Default private CacheMode cacheMode = CacheMode.NONE;
    @Builder.Default private WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default private ConvolutionLayer.AlgoMode cudnnAlgoMode = ConvolutionLayer.AlgoMode.PREFER_FASTEST;

    private VGG16() {}

    @Override
    public String pretrainedUrl(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return "http://blob.deeplearning4j.org/models/vgg16_dl4j_inference.zip";
        else if (pretrainedType == PretrainedType.CIFAR10)
            return "http://blob.deeplearning4j.org/models/vgg16_dl4j_cifar10_inference.v1.zip";
        else if (pretrainedType == PretrainedType.VGGFACE)
            return "http://blob.deeplearning4j.org/models/vgg16_dl4j_vggface_inference.v1.zip";
        else
            return null;
    }

    @Override
    public long pretrainedChecksum(PretrainedType pretrainedType) {
        if (pretrainedType == PretrainedType.IMAGENET)
            return 3501732770L;
        if (pretrainedType == PretrainedType.CIFAR10)
            return 2192260131L;
        if (pretrainedType == PretrainedType.VGGFACE)
            return 2706403553L;
        else
            return 0L;
    }

    @Override
    public Class<? extends Model> modelType() {
        return MultiLayerNetwork.class;
    }

    public MultiLayerConfiguration conf() {
        MultiLayerConfiguration conf =
                        new NeuralNetConfiguration.Builder().seed(seed)
                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                        .updater(updater)
                                        .activation(Activation.RELU)
                                        .cacheMode(cacheMode)
                                        .trainingWorkspaceMode(workspaceMode)
                                        .inferenceWorkspaceMode(workspaceMode)
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
                                        .layer(9, new SubsamplingLayer.Builder()
                                                        .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                                        .stride(2, 2).build())
                                        // block 4
                                        .layer(10, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(11, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(12, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(13, new SubsamplingLayer.Builder()
                                                        .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                                        .stride(2, 2).build())
                                        // block 5
                                        .layer(14, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(15, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(16, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(1, 1)
                                                        .padding(1, 1).nOut(512).cudnnAlgoMode(cudnnAlgoMode).build())
                                        .layer(17, new SubsamplingLayer.Builder()
                                                        .poolingType(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                                        .stride(2, 2).build())
                                        //                .layer(18, new DenseLayer.Builder().nOut(4096).dropOut(0.5)
                                        //                        .build())
                                        //                .layer(19, new DenseLayer.Builder().nOut(4096).dropOut(0.5)
                                        //                        .build())
                                        .layer(18, new OutputLayer.Builder(
                                                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).name("output")
                                                                        .nOut(numClasses).activation(Activation.SOFTMAX) // radial basis function required
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
