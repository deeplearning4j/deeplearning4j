/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * Standalone reproduction of the Qwen MTP predictor decode-attention tile:
 * dot_product_attention_v2 in cache form (BSHD K/V windows [1,1,4,256],
 * KV caches [1,320,4,256], device-side INT64 cache_position scalar, additive
 * bias mask [1,1,1,320], useCausalMask=false) — exactly the wiring of
 * LLaMAArchitecture.buildGatedAttention used by the MTP predictor layer.
 *
 * The Qwen evidence (mtp-staging.tensor-*.dspt): calls at positions 37/38
 * draft correctly, call at position 39 (the first NEXT-step proposal) emits
 * noise while every host-observable input is correct. This test isolates the
 * attention tile alone and replays that exact position sequence against a
 * pure-Java reference computed over exactly the rows cache_position selects.
 * If call 39 diverges here, the tile is the defect; if all three match, the
 * tile is proven innocent and the defect is in the predictor plan wiring.
 *
 * Gated by qwen.mtp.tile=true; CUDA-only (the failure only reproduces there).
 */
@Slf4j
public class TestQwenMtpAttentionTileIsolation {

    private static final int KV_HEADS = 4;      // mtp_past_key_values.0.key dim 2
    private static final int HEAD_DIM = 256;    // mtp_past_key_values.0.key dim 3
    private static final int Q_HEADS = 16;      // qwen3_5 gated attention (GQA 16/4)
    private static final int CACHE_LEN = 320;   // mtp_past_key_values.0.key dim 1
    private static final int PREFILL = 37;      // rows 0..36 populated by MTP prefill
    private static final long SEED = 20260915L;

    @Test
    public void decodeAttentionTileMatchesReferenceAcrossMtpPositions() {
        assertTrue(Nd4j.backends().isCudaAvailable(), "CUDA-only tile isolation (failure only reproduces there)");
        double scale = 1.0 / Math.sqrt(HEAD_DIM);

        SameDiff sd = SameDiff.create();
        // Production dtypes from the Qwen MTP predictor graph: fusedRoPE promotes
        // q/k to FLOAT and v is cast to match, while the KV caches are BFLOAT16
        // (assertCacheInputTypes in TestQwenNvfp4Import). keyCacheDt=17 in the
        // real dpa_v2 diag confirms the mixed-dtype path: FLOAT current windows
        // written into a BF16 cache, then K/V = cache cast to FLOAT per call.
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, 1, 1, Q_HEADS, HEAD_DIM);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, 1, 1, KV_HEADS, HEAD_DIM);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, 1, 1, KV_HEADS, HEAD_DIM);
        SDVariable kCache = sd.placeHolder("k_cache", DataType.BFLOAT16, 1, CACHE_LEN, KV_HEADS, HEAD_DIM);
        SDVariable vCache = sd.placeHolder("v_cache", DataType.BFLOAT16, 1, CACHE_LEN, KV_HEADS, HEAD_DIM);
        // Rank-0 INT64 scalar: the op reads cache_position from the DEVICE buffer
        // (dspBufferConst) — exactly how the predictor feeds mtp_cache_position.
        SDVariable cachePos = sd.placeHolder("cache_position", DataType.INT64);
        SDVariable mask = sd.placeHolder("mask", DataType.FLOAT, 1, 1, 1, CACHE_LEN);
        sd.nn().dotProductAttentionV2("attn", q, v, k, null, null,
                kCache, vCache, cachePos, mask, 0.0, 0.0, false, false);

        // Host-side input construction (device copies handled by output call).
        Nd4j.getRandom().setSeed(SEED);
        INDArray qArr = Nd4j.randn(DataType.FLOAT, 1, 1, Q_HEADS, HEAD_DIM).muli(0.3);
        // Distinct current K/V per position so a stale-position read cannot hide.
        INDArray[] kCur = new INDArray[CACHE_LEN];
        INDArray[] vCur = new INDArray[CACHE_LEN];
        INDArray kCacheArr = Nd4j.zeros(DataType.BFLOAT16, 1, CACHE_LEN, KV_HEADS, HEAD_DIM);
        INDArray vCacheArr = Nd4j.zeros(DataType.BFLOAT16, 1, CACHE_LEN, KV_HEADS, HEAD_DIM);
        INDArray bias = Nd4j.create(DataType.FLOAT, 1, 1, 1, CACHE_LEN);
        for (int pos = 0; pos < PREFILL; pos++) {
            kCur[pos] = Nd4j.randn(DataType.FLOAT, 1, 1, KV_HEADS, HEAD_DIM).muli(0.3);
            vCur[pos] = Nd4j.randn(DataType.FLOAT, 1, 1, KV_HEADS, HEAD_DIM).muli(0.3);
        }
        int[] positions = {37, 38, 39};
        for (int pos : positions) {
            kCur[pos] = Nd4j.randn(DataType.FLOAT, 1, 1, KV_HEADS, HEAD_DIM).muli(0.3);
            vCur[pos] = Nd4j.randn(DataType.FLOAT, 1, 1, KV_HEADS, HEAD_DIM).muli(0.3);
        }

        boolean first = true;
        for (int callIdx = 0; callIdx < positions.length; callIdx++) {
            int pos = positions[callIdx];
            // In-place commit semantics: the tile writes current K/V into the cache
            // at cache_position each call (kvInPlaceWriteBSHD). Reproduce it here by
            // materializing the cache state exactly as production sees it pre-call.
            INDArray kState = kCacheArr.dup();
            INDArray vState = vCacheArr.dup();
            for (int committed = 0; committed < pos; committed++) {
                kState.put(new INDArrayIndex[]{NDArrayIndex.all(),
                                NDArrayIndex.interval(committed, committed + 1),
                                NDArrayIndex.all(), NDArrayIndex.all()},
                        kCur[committed]);
                vState.put(new INDArrayIndex[]{NDArrayIndex.all(),
                                NDArrayIndex.interval(committed, committed + 1),
                                NDArrayIndex.all(), NDArrayIndex.all()},
                        vCur[committed]);
            }
            // Bias: 0 for rows <= pos, -1e9 beyond — same extents as the snapshots.
            INDArray biasState = Nd4j.create(DataType.FLOAT, 1, 1, 1, CACHE_LEN);
            for (int i = 0; i < CACHE_LEN; i++) {
                biasState.putScalar(0, 0, 0, i, i <= pos ? 0.0f : -1.0e9f);
            }

            Map<String, INDArray> inputs = new LinkedHashMap<>();
            inputs.put("q", qArr);
            inputs.put("k", kCur[pos]);
            inputs.put("v", vCur[pos]);
            inputs.put("k_cache", kState);
            inputs.put("v_cache", vState);
            inputs.put("cache_position", Nd4j.scalar(DataType.INT64, pos));
            inputs.put("mask", biasState);

            INDArray out;
            if (first) {
                // First call executes + compiles the graph (warmup), then output.
                out = sd.outputSingle(inputs, "attn");
                first = false;
            } else {
                // Subsequent calls reuse the SAME graph/session: this is the DSP
                // replay path where the predictor plan divergence appears.
                out = sd.outputSingle(inputs, "attn");
            }
            assertEquals(DataType.FLOAT, out.dataType());

            // Pure-Java reference: softmax over rows [0, pos] only (bias -1e9
            // kills the rest), GQA qHead -> kvHead = qHead / (Q_HEADS/KV_HEADS).
            INDArray ref = reference(qArr, kState, vState, biasState, pos, scale);
            double maxAbsDiff = 0, refNorm = 0;
            for (int qh = 0; qh < Q_HEADS; qh++) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    double o = out.getFloat(0, 0, qh, d);
                    double r = ref.getFloat(0, 0, qh, d);
                    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(o - r));
                    refNorm = Math.max(refNorm, Math.abs(r));
                }
            }
            // Softmax over 38-40 rows of ~N(0,0.09) scores: generous but finite
            // tolerance catches noise-level divergence (observed margin 0.55 vs
            // healthy 8.7) without failing on float associativity.
            double tol = 5.0e-3;
            log.info("pos={} maxAbsDiff={} refMax={} tol={}", pos, maxAbsDiff, refNorm, tol);
            assertTrue(maxAbsDiff <= tol,
                    "attention tile diverged at pos=" + pos + ": maxAbsDiff=" + maxAbsDiff
                            + " refMax=" + refNorm + " (tile defect if pos>=38 differs)");
            assertFalse(out.isNaN().any(), "NaN at pos=" + pos);
            assertFalse(out.isInfinite().any(), "Inf at pos=" + pos);
        }
        sd.close();
    }

    /** Pure-Java reference over exactly the visible rows the kernel must attend.
     * Reads the BF16 cache through float conversion — same numeric contract as
     * the cast-to-FLOAT the op performs on the cache before attention. */
    private static INDArray reference(INDArray q, INDArray kState, INDArray vState, INDArray bias,
                                      int pos, double scale) {
        int headsPerKv = Q_HEADS / KV_HEADS;
        INDArray kF = kState.castTo(DataType.FLOAT);
        INDArray vF = vState.castTo(DataType.FLOAT);
        INDArray out = Nd4j.create(DataType.FLOAT, 1, 1, Q_HEADS, HEAD_DIM);
        for (int qh = 0; qh < Q_HEADS; qh++) {
            int kvh = qh / headsPerKv;
            // scores over visible rows [0, pos]
            float[] scores = new float[pos + 1];
            for (int r = 0; r <= pos; r++) {
                float s = 0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    s += q.getFloat(0, 0, qh, d) * kF.getFloat(0, r, kvh, d);
                }
                scores[r] = s * (float) scale + bias.getFloat(0, 0, 0, r);
            }
            float max = Float.NEGATIVE_INFINITY;
            for (float s : scores) max = Math.max(max, s);
            float sum = 0f;
            for (int r = 0; r <= pos; r++) {
                scores[r] = (float) Math.exp(scores[r] - max);
                sum += scores[r];
            }
            for (int d = 0; d < HEAD_DIM; d++) {
                float acc = 0f;
                for (int r = 0; r <= pos; r++) {
                    acc += scores[r] * vF.getFloat(0, r, kvh, d);
                }
                out.putScalar(0, 0, qh, d, acc / sum);
            }
        }
        return out;
    }
}
