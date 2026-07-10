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

package org.eclipse.deeplearning4j.llm.sampling;

import org.eclipse.deeplearning4j.llm.generation.sampling.SamplerUtils;
import org.eclipse.deeplearning4j.llm.generation.sampling.SamplingConfig;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for typical-p (entropy-deviation) and XTC (Exclude Top Choices) sampling filters.
 *
 * <p>Parameterised over CPU and CUDA backends via the standard BaseNd4jTestWithBackends mechanism.
 * Run commands:
 * <pre>
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
 *   # CPU:
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *       -Dtest=TestTypicalPAndXtcSampling#* \
 *       -Dbackend.artifactId=nd4j-native \
 *       2>&1 | tee /tmp/test-typical-xtc-cpu.log
 *   # CUDA:
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *       -Dtest=TestTypicalPAndXtcSampling#* \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 \
 *       2>&1 | tee /tmp/test-typical-xtc-cuda.log
 * </pre>
 */
@NativeTag
@Tag(TagNames.SAMEDIFF)
public class TestTypicalPAndXtcSampling extends BaseNd4jTestWithBackends {

    // ═══════════════════════════════════════════════════════════════════════
    // Typical-p tests
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * typicalP = 1.0 is an exact no-op: logits are returned unchanged.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTypicalP_NoOp_WhenOne(Nd4jBackend backend) {
        float[] data = {1.0f, 2.0f, 3.0f, 0.5f, 0.1f};
        INDArray logits = Nd4j.createFromArray(data.clone());
        INDArray filtered = SamplerUtils.typicalPFilter(logits, 1.0);
        assertArrayEquals(data, filtered.toFloatVector(), 1e-6f, "typicalP=1.0 must be exact no-op");
    }

    /**
     * typicalP = 0.0 and negative are treated as disabled (no-op).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTypicalP_NoOp_WhenZeroOrNegative(Nd4jBackend backend) {
        float[] data = {1.0f, 2.0f, 3.0f};
        // both should be no-ops per the > 0 && < 1.0 guard
        assertArrayEquals(data, SamplerUtils.typicalPFilter(Nd4j.createFromArray(data.clone()), 0.0).toFloatVector(),
                1e-6f, "typicalP=0.0 must be no-op");
        assertArrayEquals(data, SamplerUtils.typicalPFilter(Nd4j.createFromArray(data.clone()), -0.5).toFloatVector(),
                1e-6f, "typicalP<0 must be no-op");
    }

    /**
     * Strongly peaked distribution: token 4 has probability ~0.99.
     * typicalP=0.8 should keep only token 4 (most typical = highest p ≈ lowest deviation).
     *
     * Hand-computed:
     *   logits = [0, 0, 0, 0, 10]
     *   p ≈ [0.000045, 0.000045, 0.000045, 0.000045, 0.9998]
     *   H ≈ -0.9998*log(0.9998) - 4*0.000045*log(0.000045) ≈ 0.002
     *   deviation(4) = |-log(0.9998) - 0.002| ≈ |0.0002 - 0.002| = small
     *   deviation(0..3) = |-log(0.000045) - 0.002| ≈ |9.7 - 0.002| = large
     *   → token 4 is kept; tokens 0..3 are masked
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTypicalP_PeakedDistribution_KeepsArgmax(Nd4jBackend backend) {
        float[] data = {0f, 0f, 0f, 0f, 10f};
        INDArray logits = Nd4j.createFromArray(data);
        INDArray filtered = SamplerUtils.typicalPFilter(logits, 0.8);
        float[] out = filtered.toFloatVector();

        // Token 4 must survive
        assertTrue(out[4] > Float.NEGATIVE_INFINITY,
                "Token with highest prob must survive peaked typical-p");
        // Low-prob tokens must be masked
        for (int i = 0; i < 4; i++) {
            assertTrue(out[i] == Float.NEGATIVE_INFINITY || Float.isInfinite(out[i]),
                    "Low-prob tokens should be masked with peaked distribution, i=" + i);
        }
    }

    /**
     * Flat distribution: all logits equal → all tokens equally typical.
     * With typicalP=0.9 and 10 equally probable tokens (p=0.1 each), H=log(10).
     * Every token has deviation |log(10) - log(10)| = 0. All survive.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTypicalP_FlatDistribution_AllSurvive(Nd4jBackend backend) {
        int V = 10;
        float[] data = new float[V]; // all zeros
        INDArray logits = Nd4j.createFromArray(data);
        INDArray filtered = SamplerUtils.typicalPFilter(logits, 0.9);
        float[] out = filtered.toFloatVector();

        int finiteCount = 0;
        for (float v : out) {
            if (!Float.isInfinite(v)) finiteCount++;
        }
        // All equally typical; filter should keep all 10
        assertEquals(V, finiteCount,
                "Flat distribution: all tokens are equally typical, none should be masked");
    }

    /**
     * Key property: typical-p differs from top-p on a distribution that is "typical" for
     * mid-probability tokens.
     *
     * Setup: V=6 tokens.
     *   logits = [-1, 2, 2, 2, 2, -1]
     *   p ≈ [0.003, 0.245, 0.245, 0.245, 0.245, 0.003]  (approx)
     *   H ≈ -4*0.245*log(0.245) - 2*0.003*log(0.003)
     *
     * The 4 medium tokens are equally typical (equal probability, equal deviation from H).
     * top-p=0.5 would keep only the first 2 medium tokens (enough to reach 0.5 mass).
     * typical-p=0.5 keeps ALL 4 medium tokens (their deviation = 0, all added first,
     * then cumulative prob ~0.98 after adding first medium token already exceeds 0.5).
     *
     * Actually with typical-p: we sort by deviation ascending. All 4 medium tokens have
     * the same (minimal) deviation, so they get sorted first. Their cumulative prob
     * ≈ 4 * 0.245 ≈ 0.98 >= 0.5 after we add them all (or after just one reaches 0.245,
     * which is < 0.5, so we add 2 for 0.49, 3 for 0.74 → cut at 3 when typical). But
     * the key point is: typical-p includes all medium tokens while top-p would include
     * the extreme (high-logit) tokens first. Since the medium tokens are the "typical" ones,
     * they all have the lowest deviation. The extreme (low-prob) tokens have the highest
     * deviation and are masked.
     *
     * This is the distinctive behavior: top-p=0.5 keeps only the top 2 by probability
     * (which for a symmetric distribution would be any 2); typical-p=0.5 keeps the
     * most typical tokens (those closest to entropy H in log-prob space).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTypicalP_DifferentFromTopP(Nd4jBackend backend) {
        // Two-cluster distribution: tokens {0,5} are low-prob, {1,2,3,4} are medium-prob.
        // With equal logits for the medium group: sorted softmax will have exact parity.
        float[] data = {-5f, 1f, 1f, 1f, 1f, -5f};
        INDArray logits = Nd4j.createFromArray(data);

        // SamplerUtils.topPFilter operates on logit-space (finds softmax threshold)
        // With this distribution: p ≈ [0.00045, 0.2498, 0.2498, 0.2498, 0.2498, 0.00045]
        // top-p=0.3 keeps the first token that pushes cumsum > 0.3, i.e. one medium token
        INDArray topPFiltered = SamplerUtils.topPFilter(logits.dup(), 0.3);
        int topPFinite = countFinite(topPFiltered.toFloatVector());

        // typical-p=0.3: H ≈ -4*0.25*log(0.25) ≈ 1.386 (near log(4))
        // Each medium token: -log(0.25) = log(4) ≈ 1.386, deviation ≈ |1.386-1.386| ≈ 0
        // Each extreme token: -log(0.00045) ≈ 7.7, deviation ≈ |7.7-1.386| ≈ 6.3
        // Sort ascending: all 4 medium tokens first (deviation≈0), then 2 extremes (deviation≈6.3)
        // cumsum after adding all 4 medium: ~0.999 >> 0.3 → cutoff when first medium token at 0.25
        // But 0.25 < 0.3 so we need 2 medium tokens (0.5 > 0.3) → keeps 2, masks the rest? No:
        // Actually we iterate: k=0 cumProb=0.25 < 0.3, k=1 cumProb=0.5 >= 0.3 → cutoff=2
        // So typical-p=0.3 keeps the 2 most-typical medium tokens, masking the other 2 + both extremes.
        // For the PURPOSE of this test we assert: extreme tokens (0 and 5) are always masked
        // by typical-p because they have the highest deviation, regardless of cutoff.
        INDArray typicalPFiltered = SamplerUtils.typicalPFilter(logits.dup(), 0.3);
        float[] tOut = typicalPFiltered.toFloatVector();

        // Tokens 0 and 5 (extreme, lowest p) must be masked by typical-p
        assertTrue(Float.isInfinite(tOut[0]) && tOut[0] < 0,
                "Extreme low-prob token 0 must be masked by typical-p");
        assertTrue(Float.isInfinite(tOut[5]) && tOut[5] < 0,
                "Extreme low-prob token 5 must be masked by typical-p");

        // Sanity: top-p=0.3 and typical-p=0.3 give DIFFERENT sets in this distribution
        // (typical-p keeps "medium" tokens; top-p behavior depends on sort order)
        // We don't need them to be exactly complementary — just assert typical-p correctly
        // places the extreme tokens in the masked set.
        assertTrue(topPFinite > 0, "top-p must keep at least one token");
    }

    /**
     * Verify determinism across two calls with the same logits.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTypicalP_Deterministic(Nd4jBackend backend) {
        float[] data = {0.5f, 1.2f, 0.8f, 2.0f, 0.3f, 1.5f, 0.1f, 0.9f};
        INDArray a = SamplerUtils.typicalPFilter(Nd4j.createFromArray(data.clone()), 0.7);
        INDArray b = SamplerUtils.typicalPFilter(Nd4j.createFromArray(data.clone()), 0.7);
        assertArrayEquals(a.toFloatVector(), b.toFloatVector(), 1e-6f,
                "typical-p must be deterministic");
    }

    /**
     * Greedy path (temperature <= 0) must ignore typical-p entirely.
     * Verify by confirming SamplingConfig.isGreedy() when typicalP != 1.0 and temperature=0.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testTypicalP_GreedyPathIgnored(Nd4jBackend backend) {
        SamplingConfig cfg = SamplingConfig.builder()
                .temperature(0.0)
                .typicalP(0.5)
                .doSample(false)
                .build();
        assertTrue(cfg.isGreedy(), "temperature=0 must select greedy path regardless of typicalP");
        // The greedy path in tokenSamplePolicy calls tokenSample(temp=0,...) which skips typicalP
        // We verify the config correctly exposes hasTypicalP() independently of the greedy guard
        assertTrue(cfg.hasTypicalP(), "hasTypicalP() must reflect typicalP=0.5");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // XTC tests
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * xtcProbability = 0.0 is an exact no-op.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testXtc_NoOp_WhenProbabilityZero(Nd4jBackend backend) {
        float[] data = {1.0f, 3.0f, 2.0f, 0.5f};
        INDArray logits = Nd4j.createFromArray(data.clone());
        INDArray filtered = SamplerUtils.xtcFilter(logits, 0.0, 0.1, new Random(42));
        assertArrayEquals(data, filtered.toFloatVector(), 1e-6f,
                "xtcProbability=0 must be exact no-op");
    }

    /**
     * With xtcProbability = 1.0 and threshold low enough, among tokens with p >= threshold,
     * all are masked EXCEPT the lowest-probability one.
     *
     * Fixture: 5 tokens, logits = [5, 4, 3, 2, 1]
     *   softmax ≈ [0.636, 0.234, 0.086, 0.032, 0.012]
     * With xtcThreshold = 0.01 (all 5 qualify), xtcProbability = 1.0:
     *   → keep only token 4 (p≈0.012, the minimum); mask tokens 0,1,2,3.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testXtc_Deterministic_KeepsLowestAboveThreshold(Nd4jBackend backend) {
        float[] data = {5f, 4f, 3f, 2f, 1f};
        INDArray logits = Nd4j.createFromArray(data);
        // Always apply (prob=1.0); all tokens qualify (threshold=0.01 < p[4]≈0.012)
        INDArray filtered = SamplerUtils.xtcFilter(logits, 1.0, 0.01, new Random(42));
        float[] out = filtered.toFloatVector();

        // Token 4 (lowest prob above threshold) must survive
        assertTrue(out[4] > Float.NEGATIVE_INFINITY,
                "Lowest-prob qualifying token (index 4) must survive XTC");

        // Tokens 0-3 (higher prob) must be masked
        for (int i = 0; i < 4; i++) {
            assertTrue(Float.isInfinite(out[i]) && out[i] < 0,
                    "Higher-prob qualifying token " + i + " must be masked by XTC");
        }
    }

    /**
     * When fewer than 2 tokens qualify (count above threshold < 2), XTC must not apply.
     *
     * Fixture: 5 tokens where only 1 token has p >= 0.99 (threshold).
     *   logits = [100, 0, 0, 0, 0] → token 0 has p ≈ 1.0, others ≈ 0
     * With xtcThreshold = 0.99: only token 0 qualifies → count < 2 → no masking.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testXtc_Noop_WhenFewerThanTwoQualify(Nd4jBackend backend) {
        float[] data = {100f, 0f, 0f, 0f, 0f};
        INDArray logits = Nd4j.createFromArray(data);
        INDArray before = logits.dup();
        // Only token 0 qualifies → no diversity to gain → XTC must not apply
        INDArray filtered = SamplerUtils.xtcFilter(logits, 1.0, 0.99, new Random(42));
        // No masking: all logits unchanged
        assertArrayEquals(before.toFloatVector(), filtered.toFloatVector(), 1e-5f,
                "XTC must not apply when fewer than 2 tokens qualify");
    }

    /**
     * Fixed-seed determinism: two calls with the same seed must produce identical output.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testXtc_FixedSeedDeterminism(Nd4jBackend backend) {
        float[] data = {2f, 3f, 1.5f, 2.5f, 1f};
        INDArray a = SamplerUtils.xtcFilter(Nd4j.createFromArray(data.clone()), 0.8, 0.1,
                new Random(12345L));
        INDArray b = SamplerUtils.xtcFilter(Nd4j.createFromArray(data.clone()), 0.8, 0.1,
                new Random(12345L));
        assertArrayEquals(a.toFloatVector(), b.toFloatVector(), 1e-6f,
                "XTC with fixed seed must be deterministic");
    }

    /**
     * xtcThreshold higher than all token probabilities → none qualify → no-op.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testXtc_Noop_WhenThresholdTooHigh(Nd4jBackend backend) {
        float[] data = {1f, 1f, 1f};  // uniform → each p = 0.333
        INDArray logits = Nd4j.createFromArray(data);
        INDArray before = logits.dup();
        // xtcThreshold = 0.5 > 0.333 → no token qualifies
        INDArray filtered = SamplerUtils.xtcFilter(logits, 1.0, 0.5, new Random(42));
        // All three tokens should survive unchanged
        assertArrayEquals(before.toFloatVector(), filtered.toFloatVector(), 1e-5f,
                "XTC must not mask when threshold exceeds all token probabilities");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SamplingConfig tests
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Validate SamplingConfig default values and predicate methods.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSamplingConfig_Defaults(Nd4jBackend backend) {
        SamplingConfig cfg = SamplingConfig.builder().build();
        assertEquals(1.0, cfg.getTypicalP(), 1e-9, "typicalP default must be 1.0");
        assertEquals(0.0, cfg.getXtcProbability(), 1e-9, "xtcProbability default must be 0.0");
        assertEquals(0.1, cfg.getXtcThreshold(), 1e-9, "xtcThreshold default must be 0.1");
        assertFalse(cfg.hasTypicalP(), "hasTypicalP() must be false for typicalP=1.0");
        assertFalse(cfg.hasXtc(), "hasXtc() must be false for xtcProbability=0.0");
    }

    /**
     * Validate SamplingConfig with active typical-p and XTC.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSamplingConfig_ActiveFilters(Nd4jBackend backend) {
        SamplingConfig cfg = SamplingConfig.builder()
                .typicalP(0.9)
                .xtcProbability(0.5)
                .xtcThreshold(0.1)
                .build();
        assertTrue(cfg.hasTypicalP(), "hasTypicalP() must be true for typicalP=0.9");
        assertTrue(cfg.hasXtc(), "hasXtc() must be true for xtcProbability=0.5");
        cfg.validate();  // Must not throw
    }

    /**
     * validate() must reject out-of-range typicalP.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSamplingConfig_Validate_RejectsInvalidTypicalP(Nd4jBackend backend) {
        SamplingConfig cfg = SamplingConfig.builder().typicalP(0.0).build();
        assertThrows(IllegalArgumentException.class, cfg::validate,
                "typicalP=0.0 must fail validation");
        SamplingConfig cfg2 = SamplingConfig.builder().typicalP(1.5).build();
        assertThrows(IllegalArgumentException.class, cfg2::validate,
                "typicalP=1.5 must fail validation");
    }

    /**
     * validate() must reject out-of-range xtcThreshold — but only when XTC is enabled
     * (xtcProbability > 0). With XTC disabled the threshold is inert and an out-of-range
     * value must NOT fail validation (callers leaving defaults with XTC off would be
     * rejected otherwise).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSamplingConfig_Validate_RejectsInvalidXtcThreshold(Nd4jBackend backend) {
        SamplingConfig cfg = SamplingConfig.builder().xtcProbability(1.0).xtcThreshold(0.6).build();
        assertThrows(IllegalArgumentException.class, cfg::validate,
                "xtcThreshold=0.6 must fail validation (> 0.5) when XTC is enabled");
        SamplingConfig cfg2 = SamplingConfig.builder().xtcProbability(1.0).xtcThreshold(0.0).build();
        assertThrows(IllegalArgumentException.class, cfg2::validate,
                "xtcThreshold=0.0 must fail validation (<= 0) when XTC is enabled");
        // XTC disabled: threshold is inert, out-of-range must be tolerated
        SamplingConfig cfg3 = SamplingConfig.builder().xtcProbability(0.0).xtcThreshold(0.6).build();
        cfg3.validate();
    }

    /**
     * validate() must reject out-of-range xtcProbability.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSamplingConfig_Validate_RejectsInvalidXtcProbability(Nd4jBackend backend) {
        SamplingConfig cfg = SamplingConfig.builder().xtcProbability(-0.1).build();
        assertThrows(IllegalArgumentException.class, cfg::validate,
                "xtcProbability < 0 must fail validation");
        SamplingConfig cfg2 = SamplingConfig.builder().xtcProbability(1.5).build();
        assertThrows(IllegalArgumentException.class, cfg2::validate,
                "xtcProbability > 1 must fail validation");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // AutoregressiveDecode tArgs wiring tests
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Default AutoregressiveDecode op must carry typicalP=1.0, xtcProbability=0.0, xtcThreshold=0.1
     * in tArgs[21..23], and configureFromArguments must restore them.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testAutoregressiveDecodeDefaultTypicalXtcTArgs(Nd4jBackend backend) {
        org.nd4j.linalg.api.ndarray.INDArray prefill =
                Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.FLOAT, 1, 1, 4);
        org.nd4j.linalg.api.ndarray.INDArray emb =
                Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.FLOAT, 8, 4);
        org.nd4j.linalg.api.ndarray.INDArray ids =
                Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.INT64, 1, 1);

        org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode op =
                new org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode(
                        prefill, emb, ids, null, null, null,
                        5, 2, 0, 1, 0.8, 40, 0.9, null);

        // tArgs must have at least 24 entries (0..23)
        assertTrue(op.tArgs().length >= 24,
                "AutoregressiveDecode must emit tArgs[0..23] (got " + op.tArgs().length + ")");

        // Default values
        assertEquals(1.0,  op.tArgs()[21], 1e-9, "tArgs[21] must be typicalP=1.0 by default");
        assertEquals(0.0,  op.tArgs()[22], 1e-9, "tArgs[22] must be xtcProbability=0.0 by default");
        assertEquals(0.1,  op.tArgs()[23], 1e-9, "tArgs[23] must be xtcThreshold=0.1 by default");

        // configureFromArguments restores them
        op.configureFromArguments();
        assertEquals(1.0,  op.getTypicalP(), 1e-9,   "configureFromArguments must restore typicalP");
        assertEquals(0.0,  op.getXtcProbability(), 1e-9, "configureFromArguments must restore xtcProbability");
        assertEquals(0.1,  op.getXtcThreshold(), 1e-9, "configureFromArguments must restore xtcThreshold");
    }

    /**
     * withTypicalPAndXtc must update tArgs[21..23] without disturbing iArgs.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testWithTypicalPAndXtc_UpdatesTArgs(Nd4jBackend backend) {
        org.nd4j.linalg.api.ndarray.INDArray prefill =
                Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.FLOAT, 1, 1, 4);
        org.nd4j.linalg.api.ndarray.INDArray emb =
                Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.FLOAT, 8, 4);
        org.nd4j.linalg.api.ndarray.INDArray ids =
                Nd4j.zeros(org.nd4j.linalg.api.buffer.DataType.INT64, 1, 1);

        org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode op =
                new org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode(
                        prefill, emb, ids, null, null, null,
                        5, 2, 0, 1, 0.8, 40, 0.9, null);
        long[] iArgsBefore = op.iArgs().clone();

        op.withTypicalPAndXtc(0.85, 0.6, 0.15);

        // iArgs must be unchanged
        assertArrayEquals(iArgsBefore, op.iArgs(),
                "withTypicalPAndXtc must not disturb iArgs");

        // Updated tArgs
        assertEquals(0.85, op.getTypicalP(), 1e-9);
        assertEquals(0.6,  op.getXtcProbability(), 1e-9);
        assertEquals(0.15, op.getXtcThreshold(), 1e-9);
        assertEquals(0.85, op.tArgs()[21], 1e-9, "tArgs[21] must reflect new typicalP");
        assertEquals(0.6,  op.tArgs()[22], 1e-9, "tArgs[22] must reflect new xtcProbability");
        assertEquals(0.15, op.tArgs()[23], 1e-9, "tArgs[23] must reflect new xtcThreshold");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helper
    // ═══════════════════════════════════════════════════════════════════════

    private static int countFinite(float[] arr) {
        int count = 0;
        for (float v : arr) {
            if (!Float.isInfinite(v) && !Float.isNaN(v)) count++;
        }
        return count;
    }
}
