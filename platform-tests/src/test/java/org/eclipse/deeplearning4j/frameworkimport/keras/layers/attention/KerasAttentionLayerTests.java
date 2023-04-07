package org.eclipse.deeplearning4j.frameworkimport.keras.layers.attention;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;

@DisplayName("Keras AttentionLayer tests")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
public class KerasAttentionLayerTests extends BaseDL4JTest {

    @Test
    @DisplayName("Keras AttentionLayer tests")
    public void testBasicDotProduct() throws Exception {
        Nd4j.getExecutioner().enableDebugMode(true);
        Nd4j.getExecutioner().enableVerboseMode(true);
        ComputationGraph computationGraph = KerasModelImport.importKerasModelAndWeights("/home/agibsonccc/Documents/GitHub/tf-test/keras-attention.h5", false);
        System.out.println(computationGraph.summary());
        INDArray input = Nd4j.rand(1,22);
        INDArray randLabels = Nd4j.rand(1,1);
        MultiDataSet dataSets = new MultiDataSet(input,randLabels);

        ComputationGraph transferLearning = new TransferLearning.GraphBuilder(computationGraph)
                .fineTuneConfiguration(new FineTuneConfiguration.Builder()
                        .updater(new Adam())

                        .build())
                .addLayer("output",new OutputLayer.Builder()
                        .activation(new ActivationSigmoid())
                        .nIn(1)
                        .nOut(1)
                        .lossFunction(LossFunctions.LossFunction.XENT)
                        .build(),"outputs")
                .setOutputs("output")
                .build();
        transferLearning.fit(dataSets);
    }

}
