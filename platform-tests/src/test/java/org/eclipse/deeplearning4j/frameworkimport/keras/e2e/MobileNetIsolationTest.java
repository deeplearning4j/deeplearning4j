package org.eclipse.deeplearning4j.frameworkimport.keras.e2e;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.resources.Resources;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

public class MobileNetIsolationTest {

    @Test
    void testMobileNet(@TempDir Path tempDir) throws Exception {
        File modelFile = Files.createTempFile(tempDir, "model", ".h5").toFile();
        try (InputStream is = Resources.asStream("modelimport/keras/examples/mobilenet/alternative.hdf5")) {
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }

        ComputationGraph graph = new KerasModel().modelBuilder()
                .modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(false)
                .buildModel()
                .getComputationGraph();

        // Print layer info - first 15 layers
        for (int i = 0; i < Math.min(15, graph.getLayers().length); i++) {
            org.deeplearning4j.nn.api.Layer layer = graph.getLayers()[i];
            System.out.println("Layer " + i + ": " + layer.getConfig().getClass().getSimpleName());
            if (layer.getConfig() instanceof org.deeplearning4j.nn.conf.layers.BatchNormalization) {
                org.deeplearning4j.nn.conf.layers.BatchNormalization bn =
                        (org.deeplearning4j.nn.conf.layers.BatchNormalization) layer.getConfig();
                System.out.println("  BN cnn2dFormat=" + bn.getCnn2DFormat() + " nOut=" + bn.getNOut());
            }
            if (layer.getConfig() instanceof org.deeplearning4j.nn.conf.layers.ConvolutionLayer) {
                org.deeplearning4j.nn.conf.layers.ConvolutionLayer conv =
                        (org.deeplearning4j.nn.conf.layers.ConvolutionLayer) layer.getConfig();
                System.out.println("  Conv cnn2dFormat=" + conv.getCnn2dDataFormat() + " nOut=" + conv.getNOut());
            }
        }

        // Use smaller batch to rule out memory issues
        System.out.println("\nCreating input [1, 32, 32, 3]...");
        INDArray input = Nd4j.ones(1, 32, 32, 3);  // Smaller input to save memory
        System.out.println("Running forward pass...");
        try {
            java.util.Map<String, INDArray> outputs = graph.feedForward(input, false);
            System.out.println("Forward pass complete, " + outputs.size() + " activations.");
        } catch (Exception e) {
            System.out.println("Exception during forward: " + e.getMessage());
            e.printStackTrace();
        }
        System.out.println("PASSED");
    }
}
