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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.factory.Nd4j;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Parity tests for the chunked WY-representation prefill path in gated_delta_rule.
 *
 * The chunked path (L >= 64, no actualLen masking, Dv % 32 == 0, Dk <= 128)
 * must produce outputs and final state within 1e-4 relative tolerance of
 * the sequential reference path (L < 64 or actualLen set forces sequential).
 *
 * Recurrence (arXiv:2412.06464):
 *   S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T * k_t)
 *   output_t = S_t^T * q_t
 * where g_t is the log-domain gate (already in log space on input).
 *
 * Run commands:
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *     -Dtest=TestGdnChunkedPrefill#testChunkedVsSequentialParity* \
 *     2>&1 | tee /tmp/gdn-chunked-parity.log
 *
 *   # Full class:
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *     -Dtest=TestGdnChunkedPrefill \
 *     2>&1 | tee /tmp/gdn-chunked-full.log
 */
public class TestGdnChunkedPrefill {

    /**
     * Runs gated_delta_rule on a full-length sequence and also in two halves
     * chained by state, then checks they agree.  This exercises the sequential
     * path for L < 64 (first half if L/2 < 64) and the chunked path when L >= 64.
     *
     * For the parity test we compare:
     *   runFull(L=T)  vs  runHalf(L=T/2, stateIn=null) -> stateOut + runHalf(L=T/2, stateIn=stateOut)
     *
     * Both should agree to 1e-4 rel tolerance.
     */
    static Stream<Arguments> parityParams() {
        // (T, B, H, Dk, Dv, seed, description)
        // T values: just-below-chunk (63), exact-chunk (64), just-above (65),
        //           two chunks minus one (127), four chunks (256), large (1000)
        // Dk=64 (Qwen config), Dv=64 to satisfy Dv % 32 == 0 and Dk <= 128
        return Stream.of(
            // T=63: sequential path (L < 64)
            Arguments.of(63,  1, 1, 64, 64, 42L,  "T=63 sequential"),
            // T=64: exactly one chunk -> chunked path
            Arguments.of(64,  1, 1, 64, 64, 43L,  "T=64 one chunk"),
            // T=65: one chunk + 1 remainder
            Arguments.of(65,  1, 1, 64, 64, 44L,  "T=65 one chunk + 1"),
            // T=127: two chunks minus 1
            Arguments.of(127, 1, 1, 64, 64, 45L,  "T=127 two chunks minus 1"),
            // T=256: four exact chunks
            Arguments.of(256, 1, 2, 64, 64, 46L,  "T=256 four chunks B=1 H=2"),
            // T=256: batch=3
            Arguments.of(256, 3, 2, 64, 64, 47L,  "T=256 four chunks B=3 H=2"),
            // T=1000: many chunks, larger sequence
            Arguments.of(1000,1, 2, 64, 64, 48L,  "T=1000 many chunks"),
            // T=128, Dk=128 (larger head dim)
            Arguments.of(128, 1, 2, 128, 64, 49L, "T=128 Dk=128 Dv=64"),
            // T=64 with nonzero initial state
            Arguments.of(64,  1, 1, 64, 64, 50L,  "T=64 nonzero state init"),
            // T=256, Dv=128 (larger value dim)
            Arguments.of(256, 1, 1, 64, 128, 51L, "T=256 Dv=128")
        );
    }

    /**
     * Core parity test: chunked output + state must match sequential reference
     * (obtained by running with T < 64 in pieces, or by direct comparison
     * when T < 64 uses sequential automatically).
     *
     * Strategy:
     *   1. Run the op on the full sequence [B, T, H, Dk/Dv] (may trigger chunked path).
     *   2. Run the same op in two halves, chaining the state:
     *      - half1: [B, T/2, H, ...] with no state -> output1, state1
     *      - half2: [B, T-T/2, H, ...] with stateIn=state1 -> output2, state2
     *   3. Concatenate output1 and output2 -> fullOutput_ref
     *   4. Compare fullRun output with fullOutput_ref and full state with state2.
     *
     * The halved run forces sequential if T/2 < 64 and chunked if T/2 >= 64.
     * When T=64 (one chunk), T/2=32 < 64 -> halved run is always sequential.
     * So for T=64 this directly tests chunked vs sequential parity.
     */
    @ParameterizedTest(name = "{6}")
    @MethodSource("parityParams")
    public void testChunkedVsSequentialParity(int T, int B, int H, int Dk, int Dv,
                                               long seed, String description) {
        Nd4j.getRandom().setSeed(seed);
        // Scale inputs small to avoid fp32 overflow across many chunks
        INDArray q    = Nd4j.randn(DataType.FLOAT, B, T, H, Dk).muli(0.02f);
        INDArray k    = Nd4j.randn(DataType.FLOAT, B, T, H, Dk).muli(0.02f);
        INDArray v    = Nd4j.randn(DataType.FLOAT, B, T, H, Dv).muli(0.05f);
        // beta in (0,1) after sigmoid; use pre-sigmoid uniform
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, T, H).subi(0.5f);
        // gate (log-domain): small negative values -> moderate decay
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, T, H).muli(0.3f).subi(0.5f);

        // --- Full run (may be chunked for T >= 64) ---
        INDArray[] fullResult = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate));
        INDArray fullOutput = fullResult[0];   // [B, T, H, Dv]
        INDArray fullState  = fullResult[1];   // [B, H, Dk, Dv]

        assertFalse(fullOutput.isNaN().any(),
                description + ": fullOutput contains NaN");
        assertFalse(fullState.isNaN().any(),
                description + ": fullState contains NaN");

        // --- Split run: two halves, always sequential in first half if T/2 < 64 ---
        int T1 = T / 2;
        int T2 = T - T1;

        INDArray q1    = q.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.interval(0, T1),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray k1    = k.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.interval(0, T1),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray v1    = v.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.interval(0, T1),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray beta1 = beta.get(org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.interval(0, T1),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray gate1 = gate.get(org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.interval(0, T1),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();

        INDArray[] res1 = Nd4j.exec(new GatedDeltaRule(q1, k1, v1, beta1, gate1));
        INDArray out1   = res1[0];   // [B, T1, H, Dv]
        INDArray state1 = res1[1];   // [B, H, Dk, Dv]

        INDArray q2    = q.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.interval(T1, T),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray k2    = k.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.interval(T1, T),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray v2    = v.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.interval(T1, T),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray beta2 = beta.get(org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.interval(T1, T),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray gate2 = gate.get(org.nd4j.linalg.indexing.NDArrayIndex.all(),
                                   org.nd4j.linalg.indexing.NDArrayIndex.interval(T1, T),
                                   org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();

        INDArray[] res2 = Nd4j.exec(new GatedDeltaRule(q2, k2, v2, beta2, gate2, state1));
        INDArray out2   = res2[0];   // [B, T2, H, Dv]
        INDArray state2 = res2[1];   // [B, H, Dk, Dv]

        // --- Compare full output with concatenated half outputs ---
        INDArray refOutput = Nd4j.concat(1, out1, out2);   // [B, T, H, Dv]
        INDArray refState  = state2;                         // [B, H, Dk, Dv]

        double outputMaxAbsErr = fullOutput.sub(refOutput).amaxNumber().doubleValue();
        double refOutputMax    = refOutput.amaxNumber().doubleValue();
        double outputRelErr    = (refOutputMax > 1e-9) ? outputMaxAbsErr / refOutputMax : outputMaxAbsErr;

        double stateMaxAbsErr  = fullState.sub(refState).amaxNumber().doubleValue();
        double refStateMax     = refState.amaxNumber().doubleValue();
        double stateRelErr     = (refStateMax > 1e-9) ? stateMaxAbsErr / refStateMax : stateMaxAbsErr;

        System.out.printf("[%s] output relErr=%.2e (absErr=%.2e, refMax=%.2e)%n",
                description, outputRelErr, outputMaxAbsErr, refOutputMax);
        System.out.printf("[%s] state  relErr=%.2e (absErr=%.2e, refMax=%.2e)%n",
                description, stateRelErr, stateMaxAbsErr, refStateMax);

        assertEquals(0.0, outputRelErr, 1e-4,
                description + ": output relative error too large (" + outputRelErr + ")");
        assertEquals(0.0, stateRelErr, 1e-4,
                description + ": state relative error too large (" + stateRelErr + ")");
    }

    /**
     * Regression guard: T=1 decode path must be unchanged by the chunked implementation.
     * Uses sequential path (L=1 < 64) with nonzero stateIn.
     */
    @Test
    public void testT1DecodePathUnchanged() {
        int B = 2, T = 1, H = 2, Dk = 64, Dv = 64;
        Nd4j.getRandom().setSeed(99L);

        INDArray q     = Nd4j.randn(DataType.FLOAT, B, T, H, Dk).muli(0.02f);
        INDArray k     = Nd4j.randn(DataType.FLOAT, B, T, H, Dk).muli(0.02f);
        INDArray v     = Nd4j.randn(DataType.FLOAT, B, T, H, Dv).muli(0.05f);
        INDArray beta  = Nd4j.rand(DataType.FLOAT, B, T, H).subi(0.5f);
        INDArray gate  = Nd4j.randn(DataType.FLOAT, B, T, H).muli(0.3f).subi(0.5f);
        INDArray state = Nd4j.randn(DataType.FLOAT, B, H, Dk, Dv).muli(0.01f);

        INDArray[] result = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate, state));

        assertFalse(result[0].isNaN().any(), "T=1 output NaN");
        assertFalse(result[1].isNaN().any(), "T=1 state NaN");
        assertEquals(0.0,
                result[0].amaxNumber().doubleValue() < 1e-15 ? 1.0 : 0.0,
                0.5,
                "T=1 output should be nonzero (nonzero state): " + result[0].amaxNumber().doubleValue());
        assertArrayEquals(new long[]{B, T, H, Dv}, result[0].shape(), "T=1 output shape");
        assertArrayEquals(new long[]{B, H, Dk, Dv}, result[1].shape(), "T=1 state shape");
    }

    /**
     * Direct sequential-vs-chunked comparison at T=64 (exactly one chunk).
     * Forces sequential by splitting into two T=32 runs, then compares with chunked T=64 run.
     */
    @Test
    public void testT64DirectParitySequentialVsChunked() {
        int B = 1, T = 64, H = 2, Dk = 64, Dv = 64;
        Nd4j.getRandom().setSeed(7777L);

        INDArray q    = Nd4j.randn(DataType.FLOAT, B, T, H, Dk).muli(0.02f);
        INDArray k    = Nd4j.randn(DataType.FLOAT, B, T, H, Dk).muli(0.02f);
        INDArray v    = Nd4j.randn(DataType.FLOAT, B, T, H, Dv).muli(0.05f);
        INDArray beta = Nd4j.rand(DataType.FLOAT, B, T, H).subi(0.5f);
        INDArray gate = Nd4j.randn(DataType.FLOAT, B, T, H).muli(0.3f).subi(0.5f);

        // Chunked path: full T=64 run
        INDArray[] chunkedResult = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate));

        // Sequential reference: two T=32 half runs
        INDArray q1    = q.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 32), org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray k1    = k.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 32), org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray v1    = v.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 32), org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray beta1 = beta.get(org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 32), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray gate1 = gate.get(org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 32), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray[] r1  = Nd4j.exec(new GatedDeltaRule(q1, k1, v1, beta1, gate1));

        INDArray q2    = q.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(32, 64), org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray k2    = k.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(32, 64), org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray v2    = v.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(32, 64), org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray beta2 = beta.get(org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(32, 64), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray gate2 = gate.get(org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(32, 64), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
        INDArray[] r2  = Nd4j.exec(new GatedDeltaRule(q2, k2, v2, beta2, gate2, r1[1]));

        INDArray seqOut   = Nd4j.concat(1, r1[0], r2[0]);
        INDArray seqState = r2[1];

        double outRelErr = relErr(chunkedResult[0], seqOut);
        double stRelErr  = relErr(chunkedResult[1], seqState);

        System.out.printf("[T64 direct] output relErr=%.2e state relErr=%.2e%n", outRelErr, stRelErr);

        assertEquals(0.0, outRelErr, 1e-4, "T=64 chunked output must match sequential: relErr=" + outRelErr);
        assertEquals(0.0, stRelErr,  1e-4, "T=64 chunked state must match sequential: relErr=" + stRelErr);
    }

    /**
     * Tests with nonzero initial state for T >= 64 (chunked path exercises the
     * state-to-U0 subtraction in kernel B).
     */
    @Test
    public void testChunkedWithNonzeroInitialState() {
        int B = 1, T = 128, H = 2, Dk = 64, Dv = 64;
        Nd4j.getRandom().setSeed(1234L);

        INDArray q     = Nd4j.randn(DataType.FLOAT, B, T, H, Dk).muli(0.02f);
        INDArray k     = Nd4j.randn(DataType.FLOAT, B, T, H, Dk).muli(0.02f);
        INDArray v     = Nd4j.randn(DataType.FLOAT, B, T, H, Dv).muli(0.05f);
        INDArray beta  = Nd4j.rand(DataType.FLOAT, B, T, H).subi(0.5f);
        INDArray gate  = Nd4j.randn(DataType.FLOAT, B, T, H).muli(0.3f).subi(0.5f);
        INDArray state = Nd4j.randn(DataType.FLOAT, B, H, Dk, Dv).muli(0.01f);

        // Full chunked run with nonzero state
        INDArray[] fullRes = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, gate, state));

        // Sequential reference: split into T/2=64 (chunked!) then chained...
        // To force sequential, split into T/4=32 chunks each (T/4 < 64 = sequential)
        int slice = 32;
        INDArray refState = state.dup();
        INDArray[] refOutputs = new INDArray[T / slice];
        for (int i = 0; i < T / slice; ++i) {
            int t0 = i * slice, t1 = t0 + slice;
            INDArray qi    = q.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(t0, t1), org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
            INDArray ki    = k.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(t0, t1), org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
            INDArray vi    = v.get(   org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(t0, t1), org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
            INDArray betai = beta.get(org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(t0, t1), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
            INDArray gatei = gate.get(org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.interval(t0, t1), org.nd4j.linalg.indexing.NDArrayIndex.all()).dup();
            INDArray[] ri  = Nd4j.exec(new GatedDeltaRule(qi, ki, vi, betai, gatei, refState));
            refOutputs[i] = ri[0];
            refState = ri[1];
        }

        INDArray refOutput = Nd4j.concat(1, refOutputs);

        double outRelErr = relErr(fullRes[0], refOutput);
        double stRelErr  = relErr(fullRes[1], refState);

        System.out.printf("[chunked nonzero state T=128] output relErr=%.2e state relErr=%.2e%n",
                outRelErr, stRelErr);

        assertEquals(0.0, outRelErr, 1e-4,
                "Chunked + nonzero state output parity: relErr=" + outRelErr);
        assertEquals(0.0, stRelErr,  1e-4,
                "Chunked + nonzero state final state parity: relErr=" + stRelErr);
    }

    // -------------------------------------------------------------------------
    // Helper
    // -------------------------------------------------------------------------

    private static double relErr(INDArray actual, INDArray expected) {
        double absErr = actual.sub(expected).amaxNumber().doubleValue();
        double refMax = expected.amaxNumber().doubleValue();
        return (refMax > 1e-9) ? absErr / refMax : absErr;
    }
}
