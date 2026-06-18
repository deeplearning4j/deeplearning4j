package org.eclipse.deeplearning4j.frameworkimport.keras.e2e;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.resources.Resources;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Map;
import java.util.List;

import org.deeplearning4j.nn.modelimport.keras.Hdf5Archive;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils;

import static org.junit.jupiter.api.Assertions.*;

public class CausalConv1DIsolationTest {

    @Test
    void testCausalK2S1D1(@TempDir Path tempDir) throws Exception {
        runCausalConv1DModel(tempDir, "causal_conv1d_k2_s1_d1_cl_model.h5");
    }

    @Test
    void testCausalK2S2D1(@TempDir Path tempDir) throws Exception {
        runCausalConv1DModel(tempDir, "causal_conv1d_k2_s2_d1_cl_model.h5");
    }

    private void runCausalConv1DModel(Path tempDir, String name) throws Exception {
        String modelPath = "modelimport/keras/examples/causal_conv1d/" + name;
        String inputsOutputPath = "modelimport/keras/examples/causal_conv1d/" +
                (name.substring(0, name.length() - "model.h5".length()) + "inputs_and_outputs.h5");

        MultiLayerNetwork model;
        File modelFile = Files.createTempFile(tempDir, "model", ".h5").toFile();
        try (InputStream is = Resources.asStream(modelPath)) {
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
        KerasSequentialModel kerasModel = new KerasModel().modelBuilder()
                .modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(true)
                .buildSequential();
        model = kerasModel.getMultiLayerNetwork();

        // Check layer configuration
        for (int i = 0; i < model.getLayers().length; i++) {
            InputPreProcessor pp = model.getLayerWiseConfigurations().getInputPreProcess(i);
            System.out.println("Layer " + i + " (" + model.getLayerNames().get(i) + "): " +
                    "type=" + model.getLayer(i).getConfig().getClass().getSimpleName() +
                    ", preprocessor=" + (pp != null ? pp.getClass().getSimpleName() : "NONE"));
            Map<String, INDArray> params = model.getLayer(i).paramTable();
            for (Map.Entry<String, INDArray> e : params.entrySet()) {
                System.out.println("  param " + e.getKey() + " shape=" +
                        java.util.Arrays.toString(e.getValue().shape()) +
                        " ews=" + e.getValue().elementWiseStride());
            }
        }

        org.deeplearning4j.nn.conf.layers.Convolution1DLayer c1d =
                (org.deeplearning4j.nn.conf.layers.Convolution1DLayer) model.getLayer(0).getConfig();
        System.out.println("ConvolutionMode=" + c1d.getConvolutionMode());
        System.out.println("CNN2DFormat=" + c1d.getCnn2dDataFormat() + " RNNFormat=" + c1d.getRnnDataFormat());
        System.out.println("KernelSize=" + java.util.Arrays.toString(c1d.getKernelSize()));
        System.out.println("Stride=" + java.util.Arrays.toString(c1d.getStride()));
        System.out.println("Dilation=" + java.util.Arrays.toString(c1d.getDilation()));

        // Print weight values
        INDArray W = model.getLayer(0).paramTable().get("W");
        System.out.println("W shape=" + java.util.Arrays.toString(W.shape()) + " order=" + W.ordering());
        org.nd4j.enums.WeightsFormat wf = org.deeplearning4j.util.ConvolutionUtils.getWeightFormat(c1d.getCnn2dDataFormat());
        System.out.println("WeightsFormat=" + wf);
        INDArray w3d = org.deeplearning4j.util.Convolution1DUtils.reshapeWeightArrayOrGradientForFormat(W, wf);
        System.out.println("w3d shape=" + java.util.Arrays.toString(w3d.shape()) + " order=" + w3d.ordering());
        for (int k = 0; k < Math.min(4, w3d.size(0)); k++) {
            System.out.println("w3d[" + k + ",:,:] (kernel pos " + k + "):\n" + w3d.slice(k));
        }

        // Load test data
        File outputsFile = Files.createTempFile(tempDir, "outputs", ".h5").toFile();
        try (InputStream is = Resources.asStream(inputsOutputPath)) {
            Files.copy(is, outputsFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
        try (Hdf5Archive outputsArchive = new Hdf5Archive(outputsFile.getAbsolutePath())) {
            List<String> inputNames = (List<String>) KerasModelUtils.parseJsonString(
                    outputsArchive.readAttributeAsJson("inputs")).get("inputs");
            INDArray input = outputsArchive.readDataSet(inputNames.get(0), "inputs");
            System.out.println("Input shape: " + java.util.Arrays.toString(input.shape()) +
                    " rank=" + input.rank());

            // Print some input values
            System.out.println("Input[0,:3,:3]:");
            for (int t = 0; t < Math.min(3, input.size(1)); t++) {
                StringBuilder sb = new StringBuilder();
                for (int ch = 0; ch < Math.min(3, input.size(2)); ch++) {
                    sb.append(String.format("%.4f ", input.getDouble(0, t, ch)));
                }
                System.out.println("  [0," + t + ",:]: " + sb);
            }

            // Get expected activations
            Map<String, INDArray> activations = new java.util.HashMap<>();
            for (String layerName : outputsArchive.getDataSets("activations")) {
                activations.put(layerName, outputsArchive.readDataSet(layerName, "activations"));
            }

            // Run forward pass
            for (int i = 0; i < model.getLayers().length; i++) {
                String layerName = model.getLayerNames().get(i);
                if (activations.containsKey(layerName)) {
                    INDArray actual = model.feedForwardToLayer(i, input, false).get(i + 1);
                    INDArray expected = activations.get(layerName);
                    System.out.println("\nLayer " + layerName + ":");
                    System.out.println("  Expected shape: " + java.util.Arrays.toString(expected.shape()) +
                            " rank=" + expected.rank());
                    System.out.println("  Actual shape: " + java.util.Arrays.toString(actual.shape()) +
                            " rank=" + actual.rank());

                    // Print all values for small arrays
                    int maxT = (int) Math.min(10, expected.size(1));
                    int maxCh = (int) Math.min(8, expected.size(2));
                    for (int t = 0; t < maxT; t++) {
                        StringBuilder sb = new StringBuilder();
                        StringBuilder sb2 = new StringBuilder();
                        for (int ch = 0; ch < maxCh; ch++) {
                            sb.append(String.format("%.6f ", expected.getDouble(0, t, ch)));
                            sb2.append(String.format("%.6f ", actual.getDouble(0, t, ch)));
                        }
                        System.out.println("  Expected[0," + t + ",:]: " + sb);
                        System.out.println("  Actual  [0," + t + ",:]: " + sb2);
                    }

                    double maxDiff = expected.sub(actual.castTo(expected.dataType())).amaxNumber().doubleValue();
                    System.out.println("  Max abs diff: " + maxDiff);
                    boolean eq = expected.equalsWithEps(actual.castTo(expected.dataType()), 1e-4);
                    assertTrue(eq, "Output differs for " + layerName + " in model " + name +
                            ", max diff: " + maxDiff);
                }
            }
        }
    }
}
