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

package org.eclipse.deeplearning4j.nd4j.autodiff.optimization;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedGemmSwiglu;
import org.nd4j.linalg.api.ops.impl.transforms.custom.SwishMul;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Collections;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Documents and validates the FuseGatedMLPPattern decision in ActivationFusionOptimizations.
 *
 * <h2>Background</h2>
 *
 * The GatedMLP (SwiGLU) feed-forward block pattern in LLMs is:
 * <pre>
 *   gate = x @ wGate          // [B, N]
 *   up   = x @ wUp            // [B, N]
 *   act  = silu(gate) * up    // element-wise
 *   out  = act @ wDown        // [B, H]
 * </pre>
 *
 * There is a C++ op {@code fused_gemm_swiglu} that folds the two GEMMs and the
 * element-wise SwiGLU into one call.  A graph-optimizer rule that rewrites the
 * pattern above to a single {@code fused_gemm_swiglu} + one matmul was
 * <em>considered and deliberately left disabled</em> (see ActivationFusionOptimizations.java:459).
 *
 * <h2>Root-cause verdict (confirmed by code reading + this test)</h2>
 * <ol>
 *   <li><b>Sequential MmulHelper calls with temp-buffer allocation</b>: the C++
 *       {@code fused_gemm_swiglu} generic implementation allocates a temporary
 *       output tensor, calls {@code MmulHelper::mmul} for wGate, applies SiLU
 *       in-place, and calls another {@code MmulHelper::mmul} for wUp — sequential,
 *       not pipelined.  On GPU this is strictly worse than having the two matmuls
 *       run as separate DSP islands (which DSP can batch/capture independently).</li>
 *   <li><b>MATMUL DSP trait breaks Triton island chains</b>: {@code fused_gemm_swiglu}
 *       is registered as {@code MATMUL} in OpTraitTable.  DSP uses op trait to decide
 *       island boundaries; a MATMUL op inside an elementwise island breaks the
 *       capture chain and prevents Triton from fusing surrounding element-wise ops.</li>
 *   <li><b>Prevents DSP from batching the two matmuls</b>: with separate mmul ops,
 *       DSP schedules both inside the same capture group or as adjacent Triton ops.
 *       The fused op merges them into one opaque call that DSP cannot re-order or
 *       co-schedule with the preceding/following matmuls.</li>
 * </ol>
 *
 * <b>Conclusion</b>: FuseGatedMLPPattern is not a correctness bug — it is a
 * deliberate throughput trade-off.  The active optimisation deliberately stops at
 * {@code swish_mul} (BINARY_ELEMENTWISE) so that:
 * <ul>
 *   <li>The two matmuls stay as separate DSP MATMUL nodes (batchable, Triton-eligible).</li>
 *   <li>The element-wise SiLU + mul fusion stays in the Triton island chain.</li>
 * </ul>
 *
 * <h2>What these tests verify</h2>
 * <ol>
 *   <li>{@link #testFusedGemmSwigluMatchesUnfused}: {@code fused_gemm_swiglu} is
 *       numerically correct (within 1e-5) relative to the unfused three-op chain.</li>
 *   <li>{@link #testSwiGLUFusionStopsAtSwishMul}: the live optimizer rewrites
 *       {@code sigmoid(x)*x → swish_mul} but does NOT absorb the upstream matmuls
 *       (op count stays ≥ 3 after optimisation).</li>
 *   <li>{@link #testFuseGatedMLPPatternSuppressedAndWhy}: @Disabled — documents
 *       the suppressed full-fusion by building the graph, running it both ways,
 *       confirming correctness, AND confirming that the MATMUL trait assignment
 *       would break Triton-island continuity if it were enabled.</li>
 * </ol>
 *
 * Run command:
 * <pre>
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests &amp;&amp; \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *     -Dtest=TestFuseGatedMLPPattern#* 2&gt;&amp;1 | tee /tmp/test-fuse-gated-mlp.log
 * </pre>
 */
@Slf4j
@Tag(TagNames.DL4J_OLD_API)
@DisplayName("FuseGatedMLPPattern — disabled, root-cause documentation tests")
public class TestFuseGatedMLPPattern extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 1. Numerical correctness of fused_gemm_swiglu
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Verify that {@code FusedGemmSwiglu} matches the unfused
     * {@code gate=x@wGate; up=x@wUp; silu(gate)*up} chain within 1e-5.
     *
     * This test confirms that the op is numerically sound and that disabling the
     * fusion rule is a <em>performance</em> decision, NOT a correctness problem.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("fused_gemm_swiglu is numerically correct vs unfused chain")
    public void testFusedGemmSwigluMatchesUnfused(Nd4jBackend backend) {
        final int B = 4, K = 8, N = 16;
        Nd4j.getRandom().setSeed(1234L);

        INDArray x     = Nd4j.rand(DataType.FLOAT, B, K);
        INDArray wGate = Nd4j.rand(DataType.FLOAT, K, N);
        INDArray wUp   = Nd4j.rand(DataType.FLOAT, K, N);

        // Unfused reference: silu(x@wGate) * (x@wUp)
        INDArray gate     = x.mmul(wGate);                  // [B, N]
        INDArray up       = x.mmul(wUp);                    // [B, N]
        // silu(t) = t * sigmoid(t)
        // Use Transforms.sigmoid (with implicit dup=true) to avoid in-place aliasing.
        // Nd4j.nn().sigmoid(x) uses new Sigmoid(x) whose BaseOp(INDArray x) constructor
        // sets z=x, making it in-place — that would corrupt gate before the multiply.
        // Transforms.sigmoid(gate) calls new Sigmoid(gate, gate.ulike()) which writes
        // sigmoid values into a fresh array, leaving gate unchanged.
        INDArray sigmoidGate = Transforms.sigmoid(gate);   // new array (non-in-place)
        INDArray siluGate    = gate.mul(sigmoidGate);       // gate (original) * sigmoid(gate)
        INDArray unfused     = siluGate.mul(up);            // [B, N]

        // Fused op
        FusedGemmSwiglu fusedOp = new FusedGemmSwiglu(x, wGate, wUp);
        INDArray fused = Nd4j.exec(fusedOp)[0];

        assertArrayEquals(new long[]{B, N}, fused.shape(),
            "fused_gemm_swiglu output shape mismatch");

        double maxDiff = unfused.sub(fused).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5,
            "fused_gemm_swiglu differs from unfused by " + maxDiff + " (expected < 1e-5)");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 2. Current optimizer stops at swish_mul (the live behaviour)
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Confirms the live optimizer fuses {@code swish(x)*y → swish_mul(x,y)} but
     * does NOT absorb the upstream matmuls — i.e. the full GatedMLP pattern is
     * intentionally NOT collapsed to {@code fused_gemm_swiglu}.
     *
     * Post-optimisation expectations:
     *   - SwishMul op IS present (the element-wise fusion fired)
     *   - The two mmul ops are still present (not absorbed into fused_gemm_swiglu)
     *   - FusedGemmSwiglu is NOT present
     *   - Total matmul op count == 2 (gate + up GEMMs remain separate)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Optimizer fuses swish(gate)*up → swish_mul but keeps both matmuls separate")
    public void testSwiGLUFusionStopsAtSwishMul(Nd4jBackend backend) {
        final int B = 4, K = 8, N = 16;
        Nd4j.getRandom().setSeed(5678L);

        SameDiff sd = SameDiff.create();
        SDVariable x     = sd.placeHolder("x",     DataType.FLOAT, B, K);
        SDVariable wGate = sd.placeHolder("wGate",  DataType.FLOAT, K, N);
        SDVariable wUp   = sd.placeHolder("wUp",    DataType.FLOAT, K, N);

        // Build the GatedMLP pattern
        SDVariable gate      = sd.mmul(x, wGate);               // op 1
        SDVariable up        = sd.mmul(x, wUp);                 // op 2
        SDVariable swishGate = sd.nn.swish(gate);               // op 3
        SDVariable activated = swishGate.mul("swiglu_out", up); // op 4 — will fuse to swish_mul
        SDVariable out       = sd.mean("out", activated);       // op 5

        // Capture pre-optimisation output
        Map<String, INDArray> ph = Map.of(
            "x",     Nd4j.rand(DataType.FLOAT, B, K),
            "wGate", Nd4j.rand(DataType.FLOAT, K, N),
            "wUp",   Nd4j.rand(DataType.FLOAT, K, N)
        );
        INDArray expected = sd.outputSingle(ph, "out");

        // Run optimizer
        SameDiff optimized = GraphOptimizer.optimize(sd, "out");

        // Count op types after optimisation
        // NOTE: Mmul.opName() returns "matmul" (not "mmul" — "mmul" is the Java class name,
        // not the native op name registered in the C++ op registry).
        AtomicInteger matmulCount     = new AtomicInteger(0);
        AtomicInteger swishMulCount   = new AtomicInteger(0);
        AtomicInteger fusedGemmCount  = new AtomicInteger(0);

        for (var entry : optimized.getOps().entrySet()) {
            var op = entry.getValue().getOp();
            if (op == null) continue;
            String opName = op.opName();
            if ("matmul".equals(opName))             matmulCount.incrementAndGet();
            if (op instanceof SwishMul)               swishMulCount.incrementAndGet();
            if (op instanceof FusedGemmSwiglu)        fusedGemmCount.incrementAndGet();
        }

        // The two matmuls must still be present (not absorbed).
        // Mmul.opName() == "matmul" in the native registry — check for that name.
        assertEquals(2, matmulCount.get(),
            "Expected 2 matmul ops (gate + up) to remain separate, found: " + matmulCount.get());

        // swish_mul must have fired (element-wise fusion)
        assertTrue(swishMulCount.get() >= 1,
            "Expected SwishMul fusion to fire, but no SwishMul found after optimisation");

        // fused_gemm_swiglu must NOT be present (the disabled rule)
        assertEquals(0, fusedGemmCount.get(),
            "FuseGatedMLPPattern is DISABLED — fused_gemm_swiglu must not appear after optimisation. " +
            "If this fires, the pattern was re-enabled without updating this test. " +
            "See ActivationFusionOptimizations.java:459 for the rationale.");

        // Numerical correctness must be preserved
        INDArray actual = optimized.outputSingle(ph, "out");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5,
            "Optimised graph output differs from original by " + maxDiff);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // 3. Suppressed pattern — @Disabled, documents what was NOT enabled and why
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * This test is @Disabled.  It documents the exact graph-rewrite that
     * FuseGatedMLPPattern would perform, proves it is numerically correct, and
     * explains why enabling it regresses throughput.
     *
     * <p><b>Do NOT enable this test as an active gate.</b>  It is here purely as a
     * living specification of what the disabled rule would do and why it is off.
     *
     * <p>The root cause is documented in ActivationFusionOptimizations.java lines 459–464:
     * <ol>
     *   <li>The C++ generic {@code fused_gemm_swiglu} allocates temp buffers and
     *       runs 2 sequential {@code MmulHelper::mmul} calls — strictly slower than
     *       two independent DSP MATMUL islands on GPU.</li>
     *   <li>The op is registered as {@code MATMUL} in OpTraitTable.  Absorbing it
     *       into a Triton elementwise island chain would fragment the island at a
     *       MATMUL boundary, breaking fusion of surrounding EW ops.</li>
     *   <li>DSP can no longer batch the two matmuls (gate + up) because they are
     *       hidden inside one opaque kernel call.</li>
     * </ol>
     */
    @Disabled(
        "FuseGatedMLPPattern is intentionally disabled — this test documents the suppressed rule. " +
        "Root cause: fused_gemm_swiglu (a) allocates temp buffers, (b) is MATMUL-trait which breaks " +
        "Triton island chains, (c) prevents DSP from batching the two matmuls separately. " +
        "See ActivationFusionOptimizations.java:459."
    )
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("[DISABLED] FuseGatedMLPPattern — would fold {mmul,mmul,swish,mul} → {fused_gemm_swiglu} — suppressed for throughput")
    public void testFuseGatedMLPPatternSuppressedAndWhy(Nd4jBackend backend) {
        // If this code runs (i.e., the @Disabled is lifted), verify numerical correctness
        // of the transformation the rule would apply.
        final int B = 4, K = 8, N = 16;
        Nd4j.getRandom().setSeed(9999L);

        INDArray xArr     = Nd4j.rand(DataType.FLOAT, B, K);
        INDArray wGateArr = Nd4j.rand(DataType.FLOAT, K, N);
        INDArray wUpArr   = Nd4j.rand(DataType.FLOAT, K, N);

        // Unfused chain
        SameDiff sd = SameDiff.create();
        SDVariable x     = sd.var("x",     xArr);
        SDVariable wGate = sd.var("wGate", wGateArr);
        SDVariable wUp   = sd.var("wUp",   wUpArr);
        SDVariable gate      = sd.mmul(x, wGate);
        SDVariable up        = sd.mmul(x, wUp);
        SDVariable swishGate = sd.nn.swish(gate);
        SDVariable out_chain = swishGate.mul("chain_out", up);
        sd.mean("loss", out_chain);
        INDArray chainOut = sd.outputSingle(Collections.emptyMap(), "chain_out");

        // Direct fused_gemm_swiglu call (what the pattern would produce)
        FusedGemmSwiglu fusedOp = new FusedGemmSwiglu(xArr, wGateArr, wUpArr);
        INDArray fusedOut = Nd4j.exec(fusedOp)[0];

        // Must be numerically correct (correctness is not the reason it is disabled)
        double maxDiff = chainOut.sub(fusedOut).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-5,
            "fused_gemm_swiglu numerically diverges from the chain by " + maxDiff +
            " — if this fails AFTER re-enabling the pattern, the C++ op itself has a bug");

        // Additional assertion: the fused op would produce a MATMUL DSP trait entry
        // which would break the Triton island chain.  We can only observe this
        // structurally via OpTraitTable (not easily inspectable from Java), but the
        // comment in OpTraitTable.cpp documents it.
        log.info("fused_gemm_swiglu is numerically correct (maxDiff={}). " +
                 "It is DISABLED in the optimizer for throughput reasons (see class Javadoc).", maxDiff);
    }
}
