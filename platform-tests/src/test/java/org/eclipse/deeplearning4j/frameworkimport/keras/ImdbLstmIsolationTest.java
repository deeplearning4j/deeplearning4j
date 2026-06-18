package org.eclipse.deeplearning4j.frameworkimport.keras;

import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.resources.Resources;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class ImdbLstmIsolationTest {

    private void validateBuffer(String label, INDArray arr) {
        DataBuffer db = arr.data();
        System.out.println(label + ": shape=" + Arrays.toString(arr.shape())
                + " offset=" + arr.offset() + " length=" + arr.length()
                + " bufLen=" + db.length()
                + " closed=" + db.wasClosed()
                + " ptrAddr=0x" + Long.toHexString(db.pointer().address())
                + " rank=" + arr.rank());
        assertFalse(db.wasClosed(), label + " buffer is closed!");
        assertTrue(db.pointer().address() != 0, label + " buffer pointer is null!");
        // Verify we can actually read the first and last elements
        float first = arr.getFloat(0);
        float last = arr.getFloat(arr.length() - 1);
        System.out.println(label + ": first=" + first + " last=" + last);
    }

    @Test
    void testImdbLstmModelLoad(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_2_model.h5";

        File modelFile = tempDir.resolve("model.h5").toFile();
        try (InputStream is = Resources.asStream(modelPath)) {
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }

        KerasSequentialModel kerasModel = new KerasModel().modelBuilder()
                .modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(false)
                .buildSequential();
        MultiLayerNetwork model = kerasModel.getMultiLayerNetwork();

        System.out.println("Model loaded successfully, layers: " + model.getnLayers());

        // Validate flat params buffer
        INDArray flatParams = model.params();
        validateBuffer("flatParams", flatParams);

        // Validate each layer's params and their buffer sharing
        DataBuffer flatDB = flatParams.data();
        long flatPtrAddr = flatDB.pointer().address();
        for (int i = 0; i < model.getnLayers(); i++) {
            System.out.println("--- Layer " + i + ": " + model.getLayer(i).getClass().getSimpleName()
                    + " name=" + model.getLayerNames().get(i));
            Map<String, INDArray> params = model.getLayer(i).paramTable();
            for (Map.Entry<String, INDArray> e : params.entrySet()) {
                INDArray p = e.getValue();
                DataBuffer pdb = p.data();
                boolean sameBuffer = (pdb.pointer().address() == flatPtrAddr);
                System.out.println("  param " + e.getKey()
                        + ": shape=" + Arrays.toString(p.shape())
                        + " offset=" + p.offset() + " length=" + p.length()
                        + " bufLen=" + pdb.length()
                        + " sameAsFlatBuffer=" + sameBuffer
                        + " closed=" + pdb.wasClosed());
                // If buffer is NOT the same as flat params, that's a problem
                if (!sameBuffer) {
                    System.out.println("  WARNING: param " + e.getKey()
                            + " does NOT share the flat params buffer! ptrAddr=0x"
                            + Long.toHexString(pdb.pointer().address())
                            + " vs flatPtrAddr=0x" + Long.toHexString(flatPtrAddr));
                }
                // Verify readability
                float first = p.getFloat(0);
                float last = p.getFloat(p.length() - 1);
                System.out.println("  readable: first=" + first + " last=" + last);
            }
        }

        // Force GC to see if it causes deallocation
        System.out.println("=== Forcing GC ===");
        System.gc();
        Thread.sleep(200);
        System.gc();
        Thread.sleep(200);

        // Re-validate after GC
        System.out.println("=== After GC ===");
        validateBuffer("flatParams-afterGC", flatParams);
        for (int i = 0; i < model.getnLayers(); i++) {
            Map<String, INDArray> params = model.getLayer(i).paramTable();
            for (Map.Entry<String, INDArray> e : params.entrySet()) {
                validateBuffer("layer" + i + "." + e.getKey() + "-afterGC", e.getValue());
            }
        }

        // Load test inputs
        String inputsOutputPath = "modelimport/keras/examples/imdb_lstm/imdb_lstm_tf_keras_2_inputs_and_outputs.h5";
        File outputsFile = tempDir.resolve("outputs.h5").toFile();
        try (InputStream is = Resources.asStream(inputsOutputPath)) {
            Files.copy(is, outputsFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }

        try (org.deeplearning4j.nn.modelimport.keras.Hdf5Archive archive =
                     new org.deeplearning4j.nn.modelimport.keras.Hdf5Archive(outputsFile.getAbsolutePath())) {
            java.util.List<String> inputNames = (java.util.List<String>)
                    KerasModelUtils.parseJsonString(archive.readAttributeAsJson("inputs")).get("inputs");
            INDArray input = archive.readDataSet(inputNames.get(0), "inputs");
            System.out.println("Input shape: " + Arrays.toString(input.shape())
                    + " dtype: " + input.dataType());

            // Test each layer one at a time
            for (int i = 0; i < model.getnLayers(); i++) {
                // Validate LSTM biases BEFORE each layer forward
                if (model.getLayer(i).paramTable().containsKey("b")) {
                    INDArray biases = model.getLayer(i).paramTable().get("b");
                    validateBuffer("biases-BEFORE-layer" + i, biases);
                }

                System.out.println("=== feedForwardToLayer(" + i + ") ===");
                System.out.flush();
                try {
                    java.util.List<INDArray> activations = model.feedForwardToLayer(i, input, false);
                    INDArray output = activations.get(activations.size() - 1);
                    System.out.println("  Output shape: " + Arrays.toString(output.shape())
                            + " dtype=" + output.dataType()
                            + " min=" + output.minNumber() + " max=" + output.maxNumber());
                } catch (Exception e) {
                    System.out.println("  FAILED at layer " + i + ": " + e.getMessage());
                    e.printStackTrace();
                    // Validate flat params after failure
                    try {
                        validateBuffer("flatParams-AFTER-FAIL", flatParams);
                    } catch (Exception ex) {
                        System.out.println("flatParams validation also failed: " + ex.getMessage());
                    }
                    throw e;
                }

                // Validate LSTM biases AFTER each layer forward
                if (model.getLayer(i).paramTable().containsKey("b")) {
                    validateBuffer("biases-AFTER-layer" + i, model.getLayer(i).paramTable().get("b"));
                }
                // Also re-check flat params
                System.out.println("flatParams after layer " + i + ": closed=" + flatParams.data().wasClosed()
                        + " ptrAddr=0x" + Long.toHexString(flatParams.data().pointer().address()));
            }
        }
    }
}
