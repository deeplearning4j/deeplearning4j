package org.eclipse.deeplearning4j.frameworkimport.keras;

import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.modelimport.keras.Hdf5Archive;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModel;
import org.deeplearning4j.nn.modelimport.keras.KerasSequentialModel;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.LSTMParamInitializer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationHardSigmoid;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.resources.Resources;

import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.*;

import static org.nd4j.linalg.indexing.NDArrayIndex.*;

/**
 * Diagnostic test to trace LSTM import and forward pass issues.
 */
public class LstmImportDiagnosticTest {

    @Test
    void traceSimpleLstmImport(@TempDir Path tempDir) throws Exception {
        String modelPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_1_model.h5";
        String inputsOutputPath = "modelimport/keras/examples/simple_lstm/simple_lstm_tf_keras_1_inputs_and_outputs.h5";

        // Import model
        File modelFile = Files.createTempFile(tempDir, "model", ".h5").toFile();
        try (InputStream is = Resources.asStream(modelPath)) {
            Files.copy(is, modelFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }

        KerasSequentialModel kerasModel = new KerasModel().modelBuilder()
                .modelHdf5Filename(modelFile.getAbsolutePath())
                .enforceTrainingConfig(false)
                .buildSequential();

        // Print Keras model layers info
        System.out.println("=== KERAS MODEL LAYERS ===");
        for (KerasLayer kl : kerasModel.getLayers().values()) {
            System.out.println("Keras layer: " + kl.getLayerName() + " class=" + kl.getClass().getSimpleName()
                + " kerasMajorVersion=" + kl.getKerasMajorVersion());
            if (kl.getWeights() != null) {
                System.out.println("  Weight keys: " + kl.getWeights().keySet());
                for (Map.Entry<String, INDArray> e : kl.getWeights().entrySet()) {
                    INDArray w = e.getValue();
                    System.out.println("  " + e.getKey() + " shape=" + Arrays.toString(w.shape())
                        + " first3=" + (w.length() > 3 ? w.reshape(1, w.length()).get(all(), interval(0, 3)) : w));
                }
            }
        }

        MultiLayerNetwork model = kerasModel.getMultiLayerNetwork();

        // Print model info
        System.out.println("\n=== MODEL STRUCTURE ===");
        LSTM lstmConf = null;
        int lstmLayerIdx = -1;
        for (int i = 0; i < model.getLayers().length; i++) {
            System.out.println("Layer " + i + ": " + model.getLayerNames().get(i) + " - " + model.getLayer(i).getClass().getSimpleName());
            if (model.getLayer(i).conf().getLayer() instanceof LSTM) {
                lstmConf = (LSTM) model.getLayer(i).conf().getLayer();
                lstmLayerIdx = i;
                System.out.println("  nIn=" + lstmConf.getNIn() + " nOut=" + lstmConf.getNOut());
                System.out.println("  activation=" + lstmConf.getActivationFn().getClass().getSimpleName());
                System.out.println("  gateActivation=" + lstmConf.getGateActivationFn().getClass().getSimpleName());
            }
        }

        if (lstmConf == null) {
            System.out.println("No LSTM layer found!");
            return;
        }

        int nIn = (int) lstmConf.getNIn();
        int nOut = (int) lstmConf.getNOut();

        // Get DL4J weights (already reordered)
        INDArray dl4jW = model.getLayer(lstmLayerIdx).getParam(LSTMParamInitializer.INPUT_WEIGHT_KEY);
        INDArray dl4jRW = model.getLayer(lstmLayerIdx).getParam(LSTMParamInitializer.RECURRENT_WEIGHT_KEY);
        INDArray dl4jB = model.getLayer(lstmLayerIdx).getParam(LSTMParamInitializer.BIAS_KEY);

        System.out.println("\n=== DL4J WEIGHTS (after import reordering) ===");
        System.out.println("W shape=" + Arrays.toString(dl4jW.shape()));
        System.out.println("RW shape=" + Arrays.toString(dl4jRW.shape()));
        System.out.println("b shape=" + Arrays.toString(dl4jB.shape()));

        // Load inputs and expected outputs
        File outputsFile = Files.createTempFile(tempDir, "outputs", ".h5").toFile();
        try (InputStream is = Resources.asStream(inputsOutputPath)) {
            Files.copy(is, outputsFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }

        try (Hdf5Archive archive = new Hdf5Archive(outputsFile.getAbsolutePath())) {
            List<String> inputNames = (List<String>) KerasModelUtils.parseJsonString(
                    archive.readAttributeAsJson("inputs")).get("inputs");
            INDArray input = archive.readDataSet(inputNames.get(0), "inputs");
            System.out.println("\n=== INPUT ===");
            System.out.println("Input shape=" + Arrays.toString(input.shape()));

            // Get expected LSTM activation
            Map<String, INDArray> activations = new HashMap<>();
            for (String layerName : archive.getDataSets("activations")) {
                activations.put(layerName, archive.readDataSet(layerName, "activations"));
            }
            INDArray expectedLstm = activations.get("lstm_1");
            System.out.println("Expected LSTM output shape=" + Arrays.toString(expectedLstm.shape()));

            // Get DL4J output
            INDArray dl4jOutput = model.feedForwardToLayer(lstmLayerIdx, input, false).get(lstmLayerIdx + 1);
            System.out.println("DL4J LSTM output shape=" + Arrays.toString(dl4jOutput.shape()));

            // Compare first 5 samples
            System.out.println("\n=== COMPARISON (first 5 samples, first 5 features) ===");
            for (int s = 0; s < Math.min(5, (int)expectedLstm.size(0)); s++) {
                System.out.println("Sample " + s + ":");
                System.out.print("  Expected: ");
                for (int f = 0; f < Math.min(5, nOut); f++) {
                    System.out.printf("%.6f ", expectedLstm.getDouble(s, f));
                }
                System.out.println();
                System.out.print("  DL4J:     ");
                for (int f = 0; f < Math.min(5, nOut); f++) {
                    System.out.printf("%.6f ", dl4jOutput.getDouble(s, f));
                }
                System.out.println();
                System.out.print("  Diff:     ");
                for (int f = 0; f < Math.min(5, nOut); f++) {
                    System.out.printf("%.6f ", expectedLstm.getDouble(s, f) - dl4jOutput.getDouble(s, f));
                }
                System.out.println();
            }

            // Manual computation for sample 0 with DL4J weights (CFOI order)
            // Input is [batch, nIn, timesteps] = [100, 1, 1]
            float x_val = input.getFloat(0, 0, 0);
            System.out.println("\n=== MANUAL COMPUTATION sample 0 ===");
            System.out.println("x = " + x_val);

            // DL4J W is [nIn, 4*nOut] in CFOI order
            INDArray x_t = Nd4j.create(new float[]{x_val}).reshape(1, 1);

            // Compute pre-activation = x * W + b
            INDArray preact = x_t.mmul(dl4jW);
            if (dl4jB.rank() == 2) {
                preact.addiRowVector(dl4jB);
            } else {
                preact.addi(dl4jB);
            }

            // DL4J CFOI order
            INDArray c_pre = preact.get(all(), interval(0, nOut)).dup();
            INDArray f_pre = preact.get(all(), interval(nOut, 2*nOut)).dup();
            INDArray o_pre = preact.get(all(), interval(2*nOut, 3*nOut)).dup();
            INDArray i_pre = preact.get(all(), interval(3*nOut, 4*nOut)).dup();

            System.out.println("Pre-activation (CFOI):");
            System.out.printf("  C[0:3]: %.6f %.6f %.6f%n", c_pre.getFloat(0,0), c_pre.getFloat(0,1), c_pre.getFloat(0,2));
            System.out.printf("  F[0:3]: %.6f %.6f %.6f%n", f_pre.getFloat(0,0), f_pre.getFloat(0,1), f_pre.getFloat(0,2));
            System.out.printf("  O[0:3]: %.6f %.6f %.6f%n", o_pre.getFloat(0,0), o_pre.getFloat(0,1), o_pre.getFloat(0,2));
            System.out.printf("  I[0:3]: %.6f %.6f %.6f%n", i_pre.getFloat(0,0), i_pre.getFloat(0,1), i_pre.getFloat(0,2));

            // Apply activations using DL4J's configured activation
            IActivation gateAct = lstmConf.getGateActivationFn();
            new ActivationTanH().getActivation(c_pre, false);  // cell input always tanh
            gateAct.getActivation(f_pre, false);
            gateAct.getActivation(o_pre, false);
            gateAct.getActivation(i_pre, false);

            // c = f * 0 + i * c (first timestep)
            INDArray cellState = i_pre.mul(c_pre);
            INDArray tanhC = cellState.dup();
            new ActivationTanH().getActivation(tanhC, false);
            INDArray h_cfoi = o_pre.mul(tanhC);

            System.out.println("DL4J manual h (CFOI, " + gateAct.getClass().getSimpleName() + "):");
            System.out.printf("  h[0:5]: %.6f %.6f %.6f %.6f %.6f%n",
                h_cfoi.getFloat(0,0), h_cfoi.getFloat(0,1), h_cfoi.getFloat(0,2), h_cfoi.getFloat(0,3), h_cfoi.getFloat(0,4));

            // Now try Keras IFCO interpretation of the SAME DL4J weights
            // If DL4J stores as CFOI, then:
            //   DL4J slot 0 = C, slot 1 = F, slot 2 = O, slot 3 = I
            // But what if it's actually stored in Keras IFCO order (wrong assumption)?
            INDArray i2_pre = preact.get(all(), interval(0, nOut)).dup();        // treating as I
            INDArray f2_pre = preact.get(all(), interval(nOut, 2*nOut)).dup();    // treating as F
            INDArray c2_pre = preact.get(all(), interval(2*nOut, 3*nOut)).dup();  // treating as C
            INDArray o2_pre = preact.get(all(), interval(3*nOut, 4*nOut)).dup();  // treating as O

            new ActivationTanH().getActivation(c2_pre, false);
            gateAct.getActivation(f2_pre, false);
            gateAct.getActivation(o2_pre, false);
            gateAct.getActivation(i2_pre, false);

            INDArray cellState2 = i2_pre.mul(c2_pre);
            INDArray tanhC2 = cellState2.dup();
            new ActivationTanH().getActivation(tanhC2, false);
            INDArray h_ifco = o2_pre.mul(tanhC2);

            System.out.println("Manual h (IFCO interpretation, " + gateAct.getClass().getSimpleName() + "):");
            System.out.printf("  h[0:5]: %.6f %.6f %.6f %.6f %.6f%n",
                h_ifco.getFloat(0,0), h_ifco.getFloat(0,1), h_ifco.getFloat(0,2), h_ifco.getFloat(0,3), h_ifco.getFloat(0,4));

            // Also try with regular sigmoid instead of hard_sigmoid
            INDArray preact2 = x_t.mmul(dl4jW);
            if (dl4jB.rank() == 2) preact2.addiRowVector(dl4jB);
            else preact2.addi(dl4jB);

            INDArray c3_pre = preact2.get(all(), interval(0, nOut)).dup();
            INDArray f3_pre = preact2.get(all(), interval(nOut, 2*nOut)).dup();
            INDArray o3_pre = preact2.get(all(), interval(2*nOut, 3*nOut)).dup();
            INDArray i3_pre = preact2.get(all(), interval(3*nOut, 4*nOut)).dup();

            new ActivationTanH().getActivation(c3_pre, false);
            new ActivationSigmoid().getActivation(f3_pre, false);
            new ActivationSigmoid().getActivation(o3_pre, false);
            new ActivationSigmoid().getActivation(i3_pre, false);

            INDArray cellState3 = i3_pre.mul(c3_pre);
            INDArray tanhC3 = cellState3.dup();
            new ActivationTanH().getActivation(tanhC3, false);
            INDArray h_cfoi_sigmoid = o3_pre.mul(tanhC3);

            System.out.println("Manual h (CFOI, sigmoid gates):");
            System.out.printf("  h[0:5]: %.6f %.6f %.6f %.6f %.6f%n",
                h_cfoi_sigmoid.getFloat(0,0), h_cfoi_sigmoid.getFloat(0,1), h_cfoi_sigmoid.getFloat(0,2), h_cfoi_sigmoid.getFloat(0,3), h_cfoi_sigmoid.getFloat(0,4));

            // Compare expected
            System.out.println("Expected h:");
            System.out.printf("  h[0:5]: %.6f %.6f %.6f %.6f %.6f%n",
                expectedLstm.getDouble(0,0), expectedLstm.getDouble(0,1), expectedLstm.getDouble(0,2), expectedLstm.getDouble(0,3), expectedLstm.getDouble(0,4));

            // Compute RMS error for each interpretation
            double rmsCfoi = rmsDiff(h_cfoi, expectedLstm.getRow(0).reshape(1, nOut));
            double rmsIfco = rmsDiff(h_ifco, expectedLstm.getRow(0).reshape(1, nOut));
            double rmsCfoiSigmoid = rmsDiff(h_cfoi_sigmoid, expectedLstm.getRow(0).reshape(1, nOut));
            System.out.println("\nRMS error:");
            System.out.printf("  CFOI + hard_sigmoid: %.8f%n", rmsCfoi);
            System.out.printf("  IFCO + hard_sigmoid: %.8f%n", rmsIfco);
            System.out.printf("  CFOI + sigmoid:      %.8f%n", rmsCfoiSigmoid);
        }
    }

    private double rmsDiff(INDArray a, INDArray b) {
        INDArray diff = a.sub(b.castTo(a.dataType()));
        return Math.sqrt(diff.mul(diff).meanNumber().doubleValue());
    }
}
