package org.eclipse.deeplearning4j.frameworkimport.keras.e2e;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.modelimport.keras.Hdf5Archive;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils;
import org.deeplearning4j.util.ConvolutionUtils;
import org.deeplearning4j.util.Convolution1DUtils;
import org.nd4j.enums.WeightsFormat;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.resources.Resources;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv1D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class Conv1DManualTest {

    @Test
    void testDirectConv1DOp(@TempDir Path tempDir) throws Exception {
        String name = "conv1d_k2_s1_d1_cl_same_model.h5";
        String modelPath = "modelimport/keras/examples/conv1d/" + name;
        String inputsOutputPath = "modelimport/keras/examples/conv1d/" +
                (name.substring(0, name.length() - "model.h5".length()) + "inputs_and_outputs.h5");

        // Load model
        File modelFile = Files.createTempFile(tempDir, "model", ".h5").toFile();
        try (InputStream is = Resources.asStream(modelPath)) {
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
        KerasSequentialModel kerasModel = new KerasModel().modelBuilder()
                .modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(false)
                .buildSequential();
        MultiLayerNetwork model = kerasModel.getMultiLayerNetwork();

        // Get weights
        INDArray W = model.getLayer(0).paramTable().get("W");
        INDArray b = model.getLayer(0).paramTable().get("b");
        System.out.println("W shape=" + java.util.Arrays.toString(W.shape()) + " order=" + W.ordering());
        System.out.println("b shape=" + java.util.Arrays.toString(b.shape()));

        // Get weight format info
        org.deeplearning4j.nn.conf.layers.Convolution1DLayer c1d =
                (org.deeplearning4j.nn.conf.layers.Convolution1DLayer) model.getLayer(0).getConfig();
        WeightsFormat wf = ConvolutionUtils.getWeightFormat(c1d.getCnn2dDataFormat());
        System.out.println("WeightsFormat=" + wf + " CNN2DFormat=" + c1d.getCnn2dDataFormat());

        // Reshape weights to [kW, iC, oC] like the layer does
        INDArray w3d = Convolution1DUtils.reshapeWeightArrayOrGradientForFormat(W, wf);
        System.out.println("w3d shape=" + java.util.Arrays.toString(w3d.shape()));

        // Print weight values
        System.out.println("w3d values (kW=2, iC=3, oC=4):");
        for (int kw = 0; kw < 2; kw++) {
            for (int ic = 0; ic < 3; ic++) {
                StringBuilder sb = new StringBuilder();
                for (int oc = 0; oc < 4; oc++) {
                    sb.append(String.format("%.6f ", w3d.getDouble(kw, ic, oc)));
                }
                System.out.println("  w[" + kw + "," + ic + ",:] = " + sb);
            }
        }
        System.out.println("bias: " + b);

        // Load test data
        File outputsFile = Files.createTempFile(tempDir, "outputs", ".h5").toFile();
        try (InputStream is = Resources.asStream(inputsOutputPath)) {
            Files.copy(is, outputsFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
        try (Hdf5Archive outputsArchive = new Hdf5Archive(outputsFile.getAbsolutePath())) {
            List<String> inputNames = (List<String>) KerasModelUtils.parseJsonString(
                    outputsArchive.readAttributeAsJson("inputs")).get("inputs");
            INDArray input = outputsArchive.readDataSet(inputNames.get(0), "inputs");
            System.out.println("Input shape: " + java.util.Arrays.toString(input.shape()) + " order=" + input.ordering());

            // Print first 3 time steps of input
            for (int t = 0; t < 3; t++) {
                StringBuilder sb = new StringBuilder();
                for (int c = 0; c < 3; c++) {
                    sb.append(String.format("%.6f ", input.getDouble(0, t, c)));
                }
                System.out.println("  input[0," + t + ",:] = " + sb);
            }

            // Get expected output
            INDArray expected = outputsArchive.readDataSet("conv0", "activations");
            System.out.println("Expected shape: " + java.util.Arrays.toString(expected.shape()) + " order=" + expected.ordering());

            // Manually compute expected output[0,0,:] using conv formula
            // output[0, t, oc] = sum_{kw,ic} input[0, t+kw, ic] * w[kw, ic, oc] + bias[oc]
            // For t=0:
            System.out.println("\nManual computation for t=0:");
            for (int oc = 0; oc < 4; oc++) {
                double val = 0;
                for (int kw = 0; kw < 2; kw++) {
                    for (int ic = 0; ic < 3; ic++) {
                        val += input.getDouble(0, 0 + kw, ic) * w3d.getDouble(kw, ic, oc);
                    }
                }
                val += b.getDouble(oc);
                System.out.println("  output[0,0," + oc + "] = " + String.format("%.6f", val)
                        + " expected=" + String.format("%.6f", expected.getDouble(0, 0, oc)));
            }

            System.out.println("\nManual computation for t=1:");
            for (int oc = 0; oc < 4; oc++) {
                double val = 0;
                for (int kw = 0; kw < 2; kw++) {
                    for (int ic = 0; ic < 3; ic++) {
                        val += input.getDouble(0, 1 + kw, ic) * w3d.getDouble(kw, ic, oc);
                    }
                }
                val += b.getDouble(oc);
                System.out.println("  output[0,1," + oc + "] = " + String.format("%.6f", val)
                        + " expected=" + String.format("%.6f", expected.getDouble(0, 1, oc)));
            }

            // Now run the actual model
            INDArray actual = model.feedForwardToLayer(0, input, false).get(1);
            System.out.println("\nModel output for first 3 positions:");
            for (int t = 0; t < 3; t++) {
                StringBuilder sb = new StringBuilder();
                StringBuilder sb2 = new StringBuilder();
                for (int oc = 0; oc < 4; oc++) {
                    sb.append(String.format("%.6f ", expected.getDouble(0, t, oc)));
                    sb2.append(String.format("%.6f ", actual.getDouble(0, t, oc)));
                }
                System.out.println("  Expected[0," + t + ",:] = " + sb);
                System.out.println("  Actual  [0," + t + ",:] = " + sb2);
            }

            // Also try directly calling Conv1D op
            System.out.println("\n--- Direct Conv1D op test ---");
            INDArray inputNCW = input.permute(0, 2, 1).dup('c');
            System.out.println("inputNCW shape=" + java.util.Arrays.toString(inputNCW.shape()));
            INDArray w3dDup = w3d.dup('c');
            INDArray bDup = b.reshape(b.length()).dup('c');

            Conv1DConfig config = Conv1DConfig.builder()
                    .k(2).s(1).p(0).d(1)
                    .dataFormat(Conv1DConfig.NCW)
                    .paddingMode(PaddingMode.SAME)
                    .build();

            Conv1D op = new Conv1D(new INDArray[]{inputNCW, w3dDup, bDup}, null, config);
            Nd4j.exec(op);
            INDArray directOut = op.getOutputArgument(0);
            System.out.println("Direct output shape=" + java.util.Arrays.toString(directOut.shape()) + " order=" + directOut.ordering());

            // Convert NCW to NWC for comparison
            INDArray directNWC = directOut.permute(0, 2, 1);
            System.out.println("Direct NWC output first 3 positions:");
            for (int t = 0; t < 3; t++) {
                StringBuilder sb = new StringBuilder();
                for (int oc = 0; oc < 4; oc++) {
                    sb.append(String.format("%.6f ", directNWC.getDouble(0, t, oc)));
                }
                System.out.println("  Direct[0," + t + ",:] = " + sb);
            }
        }
    }
}
