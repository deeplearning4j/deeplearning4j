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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.architecture.ArchitectureRegistry;
import org.nd4j.ggml.architecture.ModelArchitecture;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MoeSharedExperts;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for MoeSharedExperts op and GraniteArchitecture registration.
 */
@Slf4j
public class TestGraniteOps {

    // ========================================================================
    // MoeSharedExperts op tests
    // ========================================================================

    @Test
    public void testMoeSharedExpertsOutputShapes() {
        int batch = 2, seqLen = 3, hiddenSize = 8, expertHidden = 8;
        int numExperts = 4, topK = 2;
        int sharedIntermediate = 16;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenSize);
        INDArray routerWeights = Nd4j.randn(DataType.FLOAT, hiddenSize, numExperts);
        INDArray expertWeights = Nd4j.randn(DataType.FLOAT, numExperts, hiddenSize, expertHidden);
        INDArray sharedGateProj = Nd4j.randn(DataType.FLOAT, hiddenSize, sharedIntermediate);
        INDArray sharedUpProj = Nd4j.randn(DataType.FLOAT, hiddenSize, sharedIntermediate);
        INDArray sharedDownProj = Nd4j.randn(DataType.FLOAT, sharedIntermediate, hiddenSize);

        INDArray[] results = Nd4j.exec(new MoeSharedExperts(
                input, routerWeights, expertWeights,
                sharedGateProj, sharedUpProj, sharedDownProj,
                numExperts, topK));

        assertEquals(3, results.length, "Should have 3 outputs");

        // Output 0: [batch, seq_len, hidden_size]
        assertArrayEquals(new long[]{batch, seqLen, hiddenSize}, results[0].shape(),
                "Output shape mismatch");

        // Output 1: router_probs [batch, seq_len, num_experts]
        assertArrayEquals(new long[]{batch, seqLen, numExperts}, results[1].shape(),
                "Router probs shape mismatch");

        // Output 2: expert_indices [batch, seq_len, top_k]
        assertArrayEquals(new long[]{batch, seqLen, topK}, results[2].shape(),
                "Expert indices shape mismatch");
    }

    @Test
    public void testMoeSharedExpertsSharedAlwaysActive() {
        int batch = 1, seqLen = 1, hiddenSize = 4;
        int numExperts = 2, topK = 1;
        int sharedIntermediate = 8;

        // Use identity-like weights for shared expert to verify it's always active
        INDArray input = Nd4j.ones(DataType.FLOAT, batch, seqLen, hiddenSize);
        INDArray routerWeights = Nd4j.zeros(DataType.FLOAT, hiddenSize, numExperts);
        INDArray expertWeights = Nd4j.zeros(DataType.FLOAT, numExperts, hiddenSize, hiddenSize);

        // Shared expert with non-zero weights
        INDArray sharedGateProj = Nd4j.randn(DataType.FLOAT, hiddenSize, sharedIntermediate).mul(0.1);
        INDArray sharedUpProj = Nd4j.randn(DataType.FLOAT, hiddenSize, sharedIntermediate).mul(0.1);
        INDArray sharedDownProj = Nd4j.randn(DataType.FLOAT, sharedIntermediate, hiddenSize).mul(0.1);

        INDArray[] results = Nd4j.exec(new MoeSharedExperts(
                input, routerWeights, expertWeights,
                sharedGateProj, sharedUpProj, sharedDownProj,
                numExperts, topK));

        INDArray output = results[0];

        // With zero expert weights, only shared expert contributes
        // Output should be non-zero because shared expert has non-zero weights
        double outputNorm = output.norm2Number().doubleValue();
        log.info("Output with zero routed experts: norm={}", outputNorm);

        // The shared expert should produce non-zero output
        // (exact values depend on SwiGLU of random weights, but norm should be > 0)
        assertTrue(outputNorm > 0, "Shared expert output should be non-zero even with zero routed experts");
    }

    @Test
    public void testMoeSharedExpertsRouterSelectsTopK() {
        int batch = 1, seqLen = 1, hiddenSize = 4;
        int numExperts = 4, topK = 2;
        int sharedIntermediate = 8;

        INDArray input = Nd4j.ones(DataType.FLOAT, batch, seqLen, hiddenSize);

        // Set router weights so expert 0 and 2 are strongly preferred
        INDArray routerWeights = Nd4j.zeros(DataType.FLOAT, hiddenSize, numExperts);
        // Make all input dimensions point to expert 0 and 2
        for (int d = 0; d < hiddenSize; d++) {
            routerWeights.putScalar(d, 0, 10.0f);  // Expert 0: high score
            routerWeights.putScalar(d, 2, 5.0f);   // Expert 2: medium score
            routerWeights.putScalar(d, 1, -10.0f);  // Expert 1: low score
            routerWeights.putScalar(d, 3, -10.0f);  // Expert 3: low score
        }

        INDArray expertWeights = Nd4j.randn(DataType.FLOAT, numExperts, hiddenSize, hiddenSize).mul(0.01);
        INDArray sharedGateProj = Nd4j.randn(DataType.FLOAT, hiddenSize, sharedIntermediate).mul(0.1);
        INDArray sharedUpProj = Nd4j.randn(DataType.FLOAT, hiddenSize, sharedIntermediate).mul(0.1);
        INDArray sharedDownProj = Nd4j.randn(DataType.FLOAT, sharedIntermediate, hiddenSize).mul(0.1);

        INDArray[] results = Nd4j.exec(new MoeSharedExperts(
                input, routerWeights, expertWeights,
                sharedGateProj, sharedUpProj, sharedDownProj,
                numExperts, topK));

        INDArray expertIndices = results[2];
        log.info("Expert indices: {}", expertIndices);

        // Top-K should select experts 0 and 2
        long idx0 = expertIndices.getLong(0, 0, 0);
        long idx1 = expertIndices.getLong(0, 0, 1);
        assertTrue(idx0 == 0 || idx1 == 0, "Expert 0 should be selected (got " + idx0 + "," + idx1 + ")");
        assertTrue(idx0 == 2 || idx1 == 2, "Expert 2 should be selected (got " + idx0 + "," + idx1 + ")");
    }

    @Test
    public void testMoeSharedExpertsRouterProbsSumToOne() {
        int batch = 2, seqLen = 2, hiddenSize = 8;
        int numExperts = 4, topK = 2;
        int sharedIntermediate = 16;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenSize);
        INDArray routerWeights = Nd4j.randn(DataType.FLOAT, hiddenSize, numExperts);
        INDArray expertWeights = Nd4j.randn(DataType.FLOAT, numExperts, hiddenSize, hiddenSize).mul(0.01);
        INDArray sharedGateProj = Nd4j.randn(DataType.FLOAT, hiddenSize, sharedIntermediate).mul(0.1);
        INDArray sharedUpProj = Nd4j.randn(DataType.FLOAT, hiddenSize, sharedIntermediate).mul(0.1);
        INDArray sharedDownProj = Nd4j.randn(DataType.FLOAT, sharedIntermediate, hiddenSize).mul(0.1);

        INDArray[] results = Nd4j.exec(new MoeSharedExperts(
                input, routerWeights, expertWeights,
                sharedGateProj, sharedUpProj, sharedDownProj,
                numExperts, topK));

        INDArray routerProbs = results[1];

        // Router probs should sum to ~1.0 along the expert dimension (softmax output)
        for (int b = 0; b < batch; b++) {
            for (int s = 0; s < seqLen; s++) {
                double sum = 0;
                for (int e = 0; e < numExperts; e++) {
                    double prob = routerProbs.getDouble(b, s, e);
                    assertTrue(prob >= 0, "Router prob should be non-negative");
                    sum += prob;
                }
                assertEquals(1.0, sum, 1e-4,
                        "Router probs should sum to 1.0 (got " + sum + ")");
            }
        }
    }

    @Test
    public void testMoeSharedExpertsSameDiff() {
        int batch = 2, seqLen = 3, hiddenSize = 8;
        int numExperts = 4, topK = 2;
        int sharedIntermediate = 16;

        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hiddenSize);
        SDVariable routerWeights = sd.var("routerWeights", Nd4j.randn(DataType.FLOAT, hiddenSize, numExperts));
        SDVariable expertWeights = sd.var("expertWeights", Nd4j.randn(DataType.FLOAT, numExperts, hiddenSize, hiddenSize).mul(0.01));
        SDVariable sharedGateProj = sd.var("sharedGateProj", Nd4j.randn(DataType.FLOAT, hiddenSize, sharedIntermediate).mul(0.1));
        SDVariable sharedUpProj = sd.var("sharedUpProj", Nd4j.randn(DataType.FLOAT, hiddenSize, sharedIntermediate).mul(0.1));
        SDVariable sharedDownProj = sd.var("sharedDownProj", Nd4j.randn(DataType.FLOAT, sharedIntermediate, hiddenSize).mul(0.1));

        SDVariable[] outputs = sd.nn.moeSharedExperts(input, routerWeights, expertWeights,
                sharedGateProj, sharedUpProj, sharedDownProj, numExperts, topK);

        assertEquals(3, outputs.length, "SameDiff should have 3 output variables");

        // Execute
        Map<String, INDArray> placeholders = new java.util.HashMap<>();
        placeholders.put("input", Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenSize));
        Map<String, INDArray> result = sd.output(placeholders, outputs[0].name());

        INDArray output = result.get(outputs[0].name());
        assertNotNull(output, "Output should not be null");
        assertArrayEquals(new long[]{batch, seqLen, hiddenSize}, output.shape());
    }

    // ========================================================================
    // GraniteArchitecture registration tests
    // ========================================================================

    @Test
    public void testGraniteArchitectureRegistered() {
        assertTrue(ArchitectureRegistry.hasArchitecture("granite"),
                "granite should be registered");
        assertTrue(ArchitectureRegistry.hasArchitecture("granitemoe"),
                "granitemoe should be registered");
        assertTrue(ArchitectureRegistry.hasArchitecture("granitemoeshared"),
                "granitemoeshared should be registered");
        assertTrue(ArchitectureRegistry.hasArchitecture("granitemoehybrid"),
                "granitemoehybrid should be registered");
    }

    @Test
    public void testGraniteArchitectureDetectable() {
        ModelArchitecture arch = ArchitectureRegistry.getArchitecture("granite");
        assertNotNull(arch, "Should find granite architecture");
        assertEquals("granite", arch.getName());
        assertTrue(arch.getSupportedVariants().contains("granitemoe"));
        assertTrue(arch.getSupportedVariants().contains("granitemoeshared"));
        assertTrue(arch.getSupportedVariants().contains("granitemoehybrid"));
    }

    @Test
    public void testGraniteArchitectureAllVariantsResolveSameHandler() {
        ModelArchitecture granite = ArchitectureRegistry.getArchitecture("granite");
        ModelArchitecture graniteMoe = ArchitectureRegistry.getArchitecture("granitemoe");
        ModelArchitecture graniteMoeShared = ArchitectureRegistry.getArchitecture("granitemoeshared");
        ModelArchitecture graniteMoeHybrid = ArchitectureRegistry.getArchitecture("granitemoehybrid");

        assertNotNull(granite);
        assertSame(granite, graniteMoe, "All variants should resolve to same handler");
        assertSame(granite, graniteMoeShared, "All variants should resolve to same handler");
        assertSame(granite, graniteMoeHybrid, "All variants should resolve to same handler");
    }
}
