package org.eclipse.deeplearning4j.frameworkimport.tensorflow;

import org.junit.jupiter.api.Test;
import org.nd4j.samediff.frameworkimport.tensorflow.importer.TensorflowFrameworkImporter;

import java.util.Collections;

public class TFSingleTest {

    @Test
    public void testSingle() {
        TensorflowFrameworkImporter tensorflowFrameworkImporter = new TensorflowFrameworkImporter();
        tensorflowFrameworkImporter.runImport("/home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests/frozen-model.pb", Collections.emptyMap(),true, false);
    }

}
