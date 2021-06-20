package org.deeplearning4j.nn.modelimport.keras.layers.pooling;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.resources.Resources;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

}
