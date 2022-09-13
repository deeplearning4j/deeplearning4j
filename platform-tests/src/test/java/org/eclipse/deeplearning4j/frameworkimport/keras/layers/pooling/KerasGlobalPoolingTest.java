package org.eclipse.deeplearning4j.frameworkimport.keras.layers.pooling;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.resources.Resources;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

@DisplayName("Keras Global Pooling Tests")
@Tag(TagNames.FILE_IO)
@Tag(TagNames.KERAS)
@NativeTag
public class KerasGlobalPoolingTest extends BaseDL4JTest {

    @Test
    public void testPoolingNWHC() throws Exception {
        String absolutePath = Resources.asFile("modelimport/keras/tfkeras/GAPError.h5").getAbsolutePath();
        ComputationGraph computationGraph = KerasModelImport.importKerasModelAndWeights(absolutePath);
        INDArray sampleInput = Nd4j.ones(1,400,128);
        INDArray[] output = computationGraph.output(sampleInput);
        assertArrayEquals(new long[]{1,400,512},output[0].shape());

    }

    @Test
    public void testCollapseDimensions() throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
        File modelPath = Resources.asFile("modelimport/keras/tfkeras/test-sequential.h5");
        MultiLayerNetwork model  =
                KerasModelImport.importKerasSequentialModelAndWeights(modelPath.getAbsolutePath());
        System.out.println(model.summary());
        List<Integer> list = Arrays.asList(1,2,3,4,5,6,7,8,9,10);
        int inputs = 10;
        INDArray features = Nd4j.create(1,inputs);
        for (int i =0 ; i < list.size(); i++) {
            features.putScalar(i, list.get(i));
        }
        System.out.println(features);
        INDArray pred = model.output(features);
        assertArrayEquals(new long[]{1,7},pred.shape());
    }


}
