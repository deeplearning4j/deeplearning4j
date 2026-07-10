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

package org.eclipse.deeplearning4j.ggml;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.config.QLoraConfig;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.peft.PeftModel;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.ggml.architecture.QuantizedLinear;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GgmlQMatMul;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end check for the runtime-quantized-matmul attach path that QLoRA depends on.
 *
 * <p>Before this was wired, GGUF architectures matmul'd packed weights plainly and never emitted
 * {@code ggml_qmatmul}, so PeftModel had nothing to attach a LoRA residual to (trainable params: 0).
 * This test asserts the two links that were broken:</p>
 * <ol>
 *   <li>{@link QuantizedLinear#matMul} on a packed INT8 weight with a {@code .__q__} companion emits
 *       a {@code ggml_qmatmul} op whose weight input is the packed variable.</li>
 *   <li>{@code PeftModel} attaches a LoRA residual to that op (non-zero trainable params) when the
 *       packed weight's name matches the target modules.</li>
 * </ol>
 */
@DisplayName("QuantizedLinear -> ggml_qmatmul -> QLoRA attach")
@Tag(TagNames.SAMEDIFF)
class TestQuantizedLinearAttach {

    private static final int QUANT_Q8_0 = 4;
    private static final int BLOCK_Q8_0 = 34;   // bytes/block
    private static final int ELEMS_Q8_0 = 32;   // elems/block

    /** A valid Q8_0 packed byte buffer for a logical [N,K] weight, as a 1-D INT8 INDArray. */
    private static INDArray packedQ8_0(int n, int k) {
        int numBlocksPerRow = k / ELEMS_Q8_0;
        byte[] packed = new byte[n * numBlocksPerRow * BLOCK_Q8_0];   // zeros are a valid Q8_0 block
        INDArray arr = Nd4j.create(DataType.INT8, packed.length);
        for (int i = 0; i < packed.length; i++) arr.putScalar(i, packed[i]);
        return arr;
    }

    @Test
    @DisplayName("logical output dimension comes from quantized metadata")
    void testLogicalOutputDimUsesQuantizedMetadata() {
        int n = 8, k = 64;
        INDArray packed = packedQ8_0(n, k);
        Map<String, INDArray> weights = new HashMap<>();
        weights.put("blk.0.attn_k.weight.__q__", Nd4j.createFromArray(new long[]{QUANT_Q8_0, n, k}));

        assertEquals(n, QuantizedLinear.logicalOutputDim(weights, "blk.0.attn_k.weight", packed),
                "Packed byte length must not be used as the logical output dimension");
        assertEquals(3, QuantizedLinear.logicalOutputDim(Collections.emptyMap(), null,
                Nd4j.zeros(DataType.FLOAT, 3, 5)),
                "Dense weights still use their leading matrix dimension");
    }

    @Test
    @DisplayName("runtime-quantized projections keep fp32 output from a half model")
    void testQuantizedLinearFloatOutputKeepsLogitsFp32() {
        int n = 8, k = 64;

        SameDiff sd = SameDiff.create();
        SDVariable packedWeight = sd.var("model.layers.0.self_attn.q_proj.weight", packedQ8_0(n, k));
        SDVariable hidden = sd.placeHolder("hidden", DataType.HALF, -1, k);

        Map<String, INDArray> weights = new HashMap<>();
        weights.put("blk.0.attn_q.weight.__q__", Nd4j.createFromArray(new long[]{QUANT_Q8_0, n, k}));

        SDVariable logits = QuantizedLinear.matMul(sd, "q_out", hidden, packedWeight,
                weights, "blk.0.attn_q.weight", DataType.HALF);

        assertEquals(DataType.FLOAT, logits.dataType(),
                "Runtime-quantized projections must keep FLOAT output so downstream QLoRA residuals do not consume saturated HALF activations");

        boolean checkedQMatMul = false;
        for (SameDiffOp op : sd.getOps().values()) {
            if (op.getOp() instanceof GgmlQMatMul) {
                long[] iArgs = ((GgmlQMatMul) op.getOp()).iArgs();
                assertNotNull(iArgs, "ggml_qmatmul iArgs must be present");
                assertTrue(iArgs.length >= 4, "ggml_qmatmul must include output dtype iArg");
                assertEquals(GgmlQMatMul.OUTPUT_FLOAT32, iArgs[3],
                        "FLOAT-output quantized projection must request fp32 kernel output");
                checkedQMatMul = true;
            }
        }
        assertTrue(checkedQMatMul, "Expected a ggml_qmatmul op for the packed projection weight");
    }

    @Test
    @DisplayName("packed weight -> ggml_qmatmul op emitted, PeftModel attaches non-zero adapters")
    void testQuantizedLinearEmitsGgmlQMatMulAndPeftAttaches() {
        int M = 4, N = 8, K = 64, rank = 4;

        SameDiff sd = SameDiff.create();
        // Packed BYTE weight registered under an HF-style name, exactly as MistralArchitecture does.
        SDVariable wq = sd.var("model.layers.0.self_attn.q_proj.weight", packedQ8_0(N, K));
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, K);

        // The .__q__ companion is keyed by the GGUF tensor name in the raw weights map.
        Map<String, INDArray> weights = new HashMap<>();
        weights.put("blk.0.attn_q.weight.__q__", Nd4j.createFromArray(new long[]{QUANT_Q8_0, N, K}));

        SDVariable out = QuantizedLinear.matMul(sd, "q_out", input, wq, weights, "blk.0.attn_q.weight", DataType.FLOAT);
        // Give the graph a loss/output so it is a well-formed model.
        SDVariable loss = out.sum("loss");
        loss.markAsLoss();
        sd.setLossVariables("loss");

        // (1) A ggml_qmatmul op must exist, consuming the packed weight as its weight input.
        boolean hasGgmlQMatMul = false;
        for (SameDiffOp op : sd.getOps().values()) {
            if (op.getOp() != null && "ggml_qmatmul".equals(op.getOp().opName())) {
                hasGgmlQMatMul = true;
                assertTrue(op.getInputsToOp().contains("model.layers.0.self_attn.q_proj.weight"),
                        "ggml_qmatmul must consume the packed weight variable as an input");
            }
        }
        assertTrue(hasGgmlQMatMul,
                "QuantizedLinear.matMul on a packed INT8 weight with a .__q__ companion must emit ggml_qmatmul");

        // (2) PeftModel must attach a QLoRA residual to the ggml_qmatmul op (non-zero trainable params).
        QLoraConfig cfg = QLoraConfig.builder()
                .r(rank).loraAlpha(rank * 2)
                .loraDataType(DataType.FLOAT)
                .targetModules(Collections.singletonList("q_proj"))
                .build();
        PeftModel peft = PeftModel.fromPretrained(sd, cfg);

        assertTrue(peft.getTrainableParameterCount() > 0,
                "PeftModel must attach LoRA adapters to the ggml_qmatmul weight (trainable params > 0)");
        // loraA [rank,K] + loraB [N,rank]
        assertEquals((long) rank * K + (long) N * rank, peft.getTrainableParameterCount(),
                "Trainable params must equal loraA[rank,K] + loraB[N,rank]");

        // The packed base weight must be frozen (CONSTANT), only adapters train.
        assertEquals(VariableType.CONSTANT,
                peft.getModel().getVariable("model.layers.0.self_attn.q_proj.weight").getVariableType(),
                "The packed quantized base weight must be frozen as CONSTANT");

        String loraBName = peft.getModel().variableNames().stream()
                .filter(n -> n.endsWith("_qlora_B"))
                .findFirst()
                .orElseThrow(() -> new AssertionError("Missing QLoRA B adapter variable"));
        Map<String, INDArray> placeholders = Collections.singletonMap("input", Nd4j.rand(DataType.FLOAT, M, K));
        INDArray loraBGradient = peft.getModel().calculateGradients(placeholders, loraBName).get(loraBName);
        assertNotNull(loraBGradient, "QLoRA B gradient must be present");
        double gradientNorm = loraBGradient.norm1Number().doubleValue();
        assertTrue(Double.isFinite(gradientNorm) && gradientNorm > 0.0,
                "QLoRA B gradient must flow through QuantizedLinear output rewiring; norm1=" + gradientNorm);
    }
}
