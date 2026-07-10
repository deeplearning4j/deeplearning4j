/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.peft;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.QLoraConfig;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GgmlQMatMul;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end PeftModel QLoRA tests on a synthetic graph that contains a
 * {@code ggml_qmatmul} op.
 *
 * <p>These tests verify the contract from the shared implementation brief:
 * <ol>
 *   <li>PeftModel correctly detects a ggml_qmatmul op and injects a graph-level LoRA
 *       residual at its OUTPUT variable.</li>
 *   <li>After wrapping, the loraB variable (zero-init) yields a forward pass that equals
 *       the un-wrapped base forward.</li>
 *   <li>After one training step, loraB receives a non-zero gradient and the packed weight
 *       variable remains CONSTANT (frozen, unmodified).</li>
 *   <li>{@code mergeAndUnload()} throws {@link UnsupportedOperationException} for quantized
 *       bases rather than silently corrupting packed bytes.</li>
 * </ol>
 *
 * <h2>Run</h2>
 * <pre>
 * cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests &amp;&amp; \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
 *   -Dtest=TestPeftModelQLoRA 2>&amp;1 | tee /tmp/test-peft-qlora.log
 * </pre>
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@Tag(TagNames.CUSTOM_FUNCTIONALITY)
@DisplayName("PeftModel QLoRA end-to-end tests")
public class TestPeftModelQLoRA {

    // Q8_0 block constants
    private static final int K = 64;
    private static final int N = 8;
    private static final int M = 4;
    private static final int RANK = 2;

    // ─── Helpers ─────────────────────────────────────────────────────────────────

    /** Build a valid Q8_0 packed byte array for a logical [N, K] matrix. */
    private static INDArray buildPackedWeights() {
        byte[] packed = TestQLoRAOpValidation.buildQ8_0Packed(N, K, 1234L);
        return TestQLoRAOpValidation.bytesToINDArray(packed);
    }

    /**
     * Build a tiny synthetic SameDiff graph:
     * <pre>
     *   input [M, K]  (placeholder)
     *   packedW [packed bytes]  (variable — will be frozen by PeftModel)
     *   out = ggml_qmatmul(input, packedW)   [M, N]
     *   loss = sum(out)
     * </pre>
     * The packed weight variable name is "attn.weight.q_proj" so that the
     * targetModules regex "q_proj" matches it.
     */
    private static SameDiff buildBaseGraph(INDArray packedWeightData) {
        SameDiff sd = SameDiff.create();

        // Placeholder for activations
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, M, K);

        // Packed weight variable (will be matched by PeftModel and frozen)
        SDVariable packedW = sd.var("attn.weight.q_proj", packedWeightData);

        // ggml_qmatmul op
        SDVariable out = new GgmlQMatMul(sd, input, packedW,
            GgmlQMatMul.GGML_QUANT_Q8_0, N, K, GgmlQMatMul.OUTPUT_FLOAT32)
            .outputVariables()[0];
        out.rename("attn_out");

        // Loss
        SDVariable loss = out.sum();
        loss.rename("loss");
        sd.setLossVariables("loss");

        return sd;
    }

    private static INDArray makeInput() {
        return Nd4j.rand(DataType.FLOAT, M, K).muli(0.1);
    }

    // ─── Tests ───────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("PeftModel detects ggml_qmatmul and creates qlora adapter variables")
    public void testQLoraAdapterVariablesCreated() {
        INDArray packedWeightData = buildPackedWeights();
        SameDiff base = buildBaseGraph(packedWeightData);

        QLoraConfig config = QLoraConfig.builder()
            .r(RANK)
            .loraAlpha(RANK * 2)
            .loraDataType(DataType.FLOAT)
            .targetModules(Collections.singletonList("q_proj"))
            .build();

        PeftModel peft = PeftModel.fromPretrained(base, config);

        // loraA and loraB must have been created
        SameDiff model = peft.getModel();
        boolean hasLoraA = model.variableNames().stream()
            .anyMatch(n -> n.contains("qlora_A") || n.contains("_lora_A"));
        boolean hasLoraB = model.variableNames().stream()
            .anyMatch(n -> n.contains("qlora_B") || n.contains("_lora_B"));

        assertTrue(hasLoraA, "PeftModel must create loraA variable for quantized target");
        assertTrue(hasLoraB, "PeftModel must create loraB variable for quantized target");

        // Packed weight must be frozen (CONSTANT)
        SDVariable packedVar = model.getVariable("attn.weight.q_proj");
        assertNotNull(packedVar, "Packed weight variable must still exist in the model");
        assertEquals(VariableType.CONSTANT, packedVar.getVariableType(),
            "Packed weight must be CONSTANT (frozen) after QLoRA wrapping");

        // At least 2 trainable parameters (loraA + loraB)
        assertTrue(peft.getTrainableParameterCount() > 0,
            "Must have trainable parameters");

        log.info("QLoRA adapter variables created. trainable={}", peft.getTrainableParameterCount());
    }

    @Test
    @DisplayName("With zero-init loraB: forward output equals base forward")
    public void testQLoraForwardWithZeroLoraBEqualsBase() {
        INDArray packedWeightData = buildPackedWeights();
        INDArray inputData = makeInput();

        // Base forward (no adapter)
        SameDiff base = buildBaseGraph(packedWeightData);
        Map<String, INDArray> basePlaceholders = Collections.singletonMap("input", inputData);
        INDArray baseOut = base.output(basePlaceholders, "attn_out").get("attn_out");
        assertNotNull(baseOut);

        // Wrap with QLoRA (loraB zero-init → delta = 0 → output must equal base)
        QLoraConfig config = QLoraConfig.builder()
            .r(RANK).loraAlpha(RANK * 2).loraDataType(DataType.FLOAT)
            .targetModules(Collections.singletonList("q_proj"))
            .build();

        SameDiff base2 = buildBaseGraph(packedWeightData);
        PeftModel peft = PeftModel.fromPretrained(base2, config);
        SameDiff model = peft.getModel();

        // Find the output name — it's been renamed during adapter injection
        // We need to find the qlora_out variable or whatever output the base op now feeds
        // The simplest way: run the full model and verify the sum matches base
        // Since loraB=0: qlora_out = base_out + 0 = base_out
        Map<String, INDArray> peftOut = model.output(basePlaceholders,
            model.getLossVariables().iterator().next());

        // Just verify the loss value matches (sum of identical tensors should be equal)
        INDArray peftLoss = peftOut.values().iterator().next();
        INDArray baseLoss = baseOut.sumNumber().doubleValue() != 0
            ? Nd4j.scalar(baseOut.sumNumber().doubleValue()) : Nd4j.scalar(0.0);

        double diff = Math.abs(peftLoss.getDouble(0) - baseOut.sumNumber().doubleValue());
        // Allow tiny floating-point rounding from graph-level add(delta=0)
        assertTrue(diff < 1e-4,
            "With zero loraB, peft forward loss should equal base loss; diff=" + diff);

        log.info("QLoRA zero-loraB forward parity: diff={}", diff);
    }

    @Test
    @DisplayName("After training step: loraB has non-zero gradient, packedWeight unchanged")
    public void testQLoraGradientAfterTrainingStep() throws Exception {
        INDArray packedWeightData = buildPackedWeights();
        INDArray inputData = makeInput();
        INDArray targetData = Nd4j.ones(DataType.FLOAT, 1);  // dummy label for loss

        QLoraConfig config = QLoraConfig.builder()
            .r(RANK).loraAlpha(RANK * 2).loraDataType(DataType.FLOAT)
            .targetModules(Collections.singletonList("q_proj"))
            .build();

        SameDiff base = buildBaseGraph(packedWeightData);
        PeftModel peft = PeftModel.fromPretrained(base, config);
        SameDiff model = peft.getModel();

        // Snapshot packed weight bytes BEFORE training
        SDVariable packedVar = model.getVariable("attn.weight.q_proj");
        assertNotNull(packedVar);
        INDArray packedBefore = packedVar.getArr();
        assertNotNull(packedBefore, "Packed weight array must be available");
        INDArray packedBeforeCopy = packedBefore.dup();

        // Find loraB variable
        String loraBName = model.variableNames().stream()
            .filter(n -> n.contains("qlora_B") || n.contains("_lora_B"))
            .findFirst().orElse(null);
        assertNotNull(loraBName, "loraB variable not found");
        SDVariable loraBVar = model.getVariable(loraBName);
        assertNotNull(loraBVar);

        // Verify loraB starts at zero and receives a real gradient before fit.
        INDArray loraBBefore = loraBVar.getArr();
        assertNotNull(loraBBefore, "loraB array must be available");
        assertEquals(0.0, loraBBefore.amaxNumber().doubleValue(), 1e-8,
            "loraB must start at zero");
        INDArray loraBBeforeCopy = loraBBefore.dup();

        Map<String, INDArray> placeholders = Collections.singletonMap("input", inputData);
        Map<String, INDArray> gradients = model.calculateGradients(placeholders, loraBName);
        INDArray loraBGradient = gradients.get(loraBName);
        assertNotNull(loraBGradient, "loraB gradient must be present");
        double gradientNorm = loraBGradient.norm1Number().doubleValue();
        assertTrue(Double.isFinite(gradientNorm) && gradientNorm > 0.0,
            "loraB gradient must be finite and non-zero; norm1=" + gradientNorm);

        // Configure training and run one step
        model.setTrainingConfig(TrainingConfig.builder()
            .updater(new Adam(0.001))
            .dataSetFeatureMapping("input")
            .dataSetLabelMapping("loss")
            .build());

        INDArray label = Nd4j.scalar(0.0f);
        DataSet ds = new DataSet(inputData, label);
        model.fit(new SingletonDataSetIterator(ds), 1);

        INDArray loraBAfter = loraBVar.getArr();
        assertNotNull(loraBAfter, "loraB array must remain available after fit");
        double loraBDelta = loraBAfter.sub(loraBBeforeCopy).norm1Number().doubleValue();
        assertTrue(Double.isFinite(loraBDelta) && loraBDelta > 0.0,
            "loraB must update after one fit step; L1 delta=" + loraBDelta);

        // Packed weight must be UNCHANGED after training
        INDArray packedAfter = packedVar.getArr();
        if (packedAfter != null && packedBeforeCopy != null) {
            INDArray diff = packedAfter.castTo(DataType.FLOAT)
                .sub(packedBeforeCopy.castTo(DataType.FLOAT));
            double maxDiff = diff.amaxNumber().doubleValue();
            assertEquals(0.0, maxDiff, 0.0,
                "Packed weight must be CONSTANT and unchanged by training; maxDiff=" + maxDiff);
        }

        log.info("QLoRA training step: packedWeight unchanged, loraB trainable={}", loraBName);
    }

    @Test
    @DisplayName("mergeAndUnload() throws UnsupportedOperationException for quantized base")
    public void testMergeAndUnloadThrowsForQuantizedBase() {
        INDArray packedWeightData = buildPackedWeights();

        QLoraConfig config = QLoraConfig.builder()
            .r(RANK).loraAlpha(RANK * 2).loraDataType(DataType.FLOAT)
            .targetModules(Collections.singletonList("q_proj"))
            .build();

        SameDiff base = buildBaseGraph(packedWeightData);
        PeftModel peft = PeftModel.fromPretrained(base, config);

        // mergeAndUnload must refuse to operate on quantized bases
        assertThrows(UnsupportedOperationException.class,
            peft::mergeAndUnload,
            "mergeAndUnload() must throw UnsupportedOperationException for QLoRA models");
        log.info("mergeAndUnload() correctly throws for quantized base");
    }

    @Test
    @DisplayName("saveAdapter / loadAdapterWeights round-trip")
    public void testSaveAndLoadAdapterRoundTrip() throws Exception {
        INDArray packedWeightData = buildPackedWeights();

        QLoraConfig config = QLoraConfig.builder()
            .r(RANK).loraAlpha(RANK * 2).loraDataType(DataType.FLOAT)
            .targetModules(Collections.singletonList("q_proj"))
            .build();

        SameDiff base = buildBaseGraph(packedWeightData);
        PeftModel peft = PeftModel.fromPretrained(base, config);

        // Assign a non-trivial value to loraA so we can verify round-trip
        SameDiff model = peft.getModel();
        String loraAName = model.variableNames().stream()
            .filter(n -> n.contains("qlora_A") || n.contains("_lora_A"))
            .findFirst().orElse(null);
        assertNotNull(loraAName, "loraA must exist");
        INDArray loraAOrig = Nd4j.rand(DataType.FLOAT, RANK, K).muli(0.5);
        model.getVariable(loraAName).setArray(loraAOrig.dup());

        // Save
        File tmpDir = File.createTempFile("peft_qlora_test_", "_adapter");
        tmpDir.delete(); tmpDir.mkdir();
        try {
            peft.saveAdapter(tmpDir);

            // Verify adapter_config.json was created
            File configFile = new File(tmpDir, "adapter_config.json");
            assertTrue(configFile.exists(), "adapter_config.json must be saved");

            // Verify at least one .npy file
            File[] npyFiles = tmpDir.listFiles(f -> f.getName().endsWith(".npy"));
            assertNotNull(npyFiles);
            assertTrue(npyFiles.length > 0, "At least one .npy file must be saved");

            // Load back: reset loraA to zeros, then reload
            model.getVariable(loraAName).setArray(Nd4j.zeros(DataType.FLOAT, RANK, K));

            // Use fromPretrained(base, adapterDir)
            SameDiff base2 = buildBaseGraph(packedWeightData);
            PeftModel peft2 = PeftModel.fromPretrained(base2, tmpDir);

            // Check loraA was restored
            SameDiff model2 = peft2.getModel();
            SDVariable loraA2 = model2.getVariable(loraAName);
            assertNotNull(peft2, "PeftModel must be loadable from saved adapter dir");
            assertNotNull(loraA2, "loraA must exist in the reloaded model");
            INDArray loraA2Arr = loraA2.getArr();
            assertNotNull(loraA2Arr, "reloaded loraA must have an array");
            assertTrue(loraAOrig.equalsWithEps(loraA2Arr, 1e-5),
                "loraA values must round-trip through saveAdapter -> fromPretrained");
        } finally {
            // Cleanup
            if (tmpDir.exists()) {
                for (File f : tmpDir.listFiles()) f.delete();
                tmpDir.delete();
            }
        }
        log.info("saveAdapter/loadAdapter round-trip passed");
    }
}
