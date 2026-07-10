package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheQuantize;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tiny eager op-level isolation of the INT8-V2 quantised decode attention.
 * No model, no DSP, no pipeline — exercises dot_product_attention_v2's INT8 branch
 * directly (quantise-on-write + fused quantised read) and compares to the float path
 * on identical data. Splits kernel correctness from DSP capture/wiring.
 */
@Slf4j
public class TestInt8AttnIsolation {

    @Test
    public void testQuantisedDecodeAttnVsFloat() {
        Nd4j.getRandom().setSeed(42);
        final int B = 1, qH = 2, kvH = 2, hd = 8, S = 4; // cache length 4, GQA 1:1 here
        final int cachePosVal = S - 1;                    // write current token at last slot
        final double scaleFactor = 1.0 / Math.sqrt(hd);

        // Decode-step tensors: 1 query token, current K/V for that token, pre-filled cache.
        INDArray query      = Nd4j.rand(DataType.FLOAT, B, 1, qH, hd);
        INDArray keyCur     = Nd4j.rand(DataType.FLOAT, B, 1, kvH, hd);
        INDArray valCur     = Nd4j.rand(DataType.FLOAT, B, 1, kvH, hd);
        INDArray keyCacheF  = Nd4j.rand(DataType.FLOAT, B, S, kvH, hd);
        INDArray valCacheF  = Nd4j.rand(DataType.FLOAT, B, S, kvH, hd);
        INDArray cachePos   = Nd4j.scalar((long) cachePosVal).castTo(DataType.INT64);
        INDArray empty      = Nd4j.empty(DataType.FLOAT);

        // ── Float reference: op writes keyCur at cachePos then attends over the cache ──
        INDArray refOut = execDpaV2(query, valCur, keyCur, keyCacheF.dup(), valCacheF.dup(),
                cachePos, scaleFactor, null, null);
        log.info("[FLOAT] out[0,0,0,:]={}", Arrays.toString(sliceHead(refOut)));

        // ── Quantise the caches per (pos,head) row over headDim (matches read-kernel scale layout) ──
        int rows = B * S * kvH;
        INDArray[] kq = quantizeCache(keyCacheF, rows, hd, B, S, kvH);
        INDArray[] vq = quantizeCache(valCacheF, rows, hd, B, S, kvH);
        INDArray keyCacheI8 = kq[0], keyScale = kq[1];
        INDArray valCacheI8 = vq[0], valScale = vq[1];

        // ── Quantised path: INT8 caches + scale inputs 9/10 ──
        INDArray quantOut = execDpaV2(query, valCur, keyCur, keyCacheI8, valCacheI8,
                cachePos, scaleFactor, keyScale, valScale);
        log.info("[QUANT] out[0,0,0,:]={}", Arrays.toString(sliceHead(quantOut)));

        // ── Compare (INT8 tolerance) ──
        INDArray diff = refOut.sub(quantOut);
        double maxAbs = Nd4j.getExecutioner().execAndReturn(
                new org.nd4j.linalg.api.ops.impl.reduce.same.AMax(diff)).getFinalResult().doubleValue();
        double refMax = refOut.amaxNumber().doubleValue();
        double rel = refMax > 1e-9 ? maxAbs / refMax : maxAbs;
        log.info("[ISOLATION] maxAbsErr={} refMax={} relErr={}", maxAbs, refMax, rel);
        assertTrue(rel < 0.10,
                "Quantised decode attn vs float relErr " + rel + " (>10%). FLOAT=" +
                        Arrays.toString(sliceHead(refOut)) + " QUANT=" + Arrays.toString(sliceHead(quantOut)));
    }

    /**
     * ADR 0107 V2 INLINE-SCALE variant: same comparison, but the scales ride in the tail of the
     * combined INT8 KV DataBuffer (no inputs 9/10) — exactly the layout GenerationPipeline feeds
     * the frozen decode plan. Exercises KVCacheQuantize inline mode + the null-scale (inline)
     * derivation in kvInPlaceWriteQuantisedBSHD and fusedGQADecodeQuantisedCuda.
     */
    @Test
    public void testQuantisedDecodeAttnInlineScaleVsFloat() {
        Nd4j.getRandom().setSeed(42);
        final int B = 1, qH = 2, kvH = 2, hd = 8, S = 4;
        final int cachePosVal = S - 1;
        final double scaleFactor = 1.0 / Math.sqrt(hd);

        INDArray query      = Nd4j.rand(DataType.FLOAT, B, 1, qH, hd);
        INDArray keyCur     = Nd4j.rand(DataType.FLOAT, B, 1, kvH, hd);
        INDArray valCur     = Nd4j.rand(DataType.FLOAT, B, 1, kvH, hd);
        INDArray keyCacheF  = Nd4j.rand(DataType.FLOAT, B, S, kvH, hd);
        INDArray valCacheF  = Nd4j.rand(DataType.FLOAT, B, S, kvH, hd);
        INDArray cachePos   = Nd4j.scalar((long) cachePosVal).castTo(DataType.INT64);

        // Float reference
        INDArray refOut = execDpaV2(query, valCur, keyCur, keyCacheF.dup(), valCacheF.dup(),
                cachePos, scaleFactor, null, null);
        log.info("[FLOAT-inline] out[0,0,0,:]={}", Arrays.toString(sliceHead(refOut)));

        // Combined inline quantize (values ++ f32 scale tail in ONE INT8 buffer) — pipeline recipe.
        INDArray keyCacheI8 = quantizeCacheInline(keyCacheF, B, S, kvH, hd);
        INDArray valCacheI8 = quantizeCacheInline(valCacheF, B, S, kvH, hd);

        // Quantised path with NO scale inputs — the op derives scales from the buffer tail.
        INDArray quantOut = execDpaV2(query, valCur, keyCur, keyCacheI8, valCacheI8,
                cachePos, scaleFactor, null, null);
        log.info("[QUANT-inline] out[0,0,0,:]={}", Arrays.toString(sliceHead(quantOut)));

        INDArray diff = refOut.sub(quantOut);
        double maxAbs = Nd4j.getExecutioner().execAndReturn(
                new org.nd4j.linalg.api.ops.impl.reduce.same.AMax(diff)).getFinalResult().doubleValue();
        double refMax = refOut.amaxNumber().doubleValue();
        double rel = refMax > 1e-9 ? maxAbs / refMax : maxAbs;
        log.info("[ISOLATION-inline] maxAbsErr={} refMax={} relErr={}", maxAbs, refMax, rel);
        assertTrue(rel < 0.10,
                "INLINE quantised decode attn vs float relErr " + rel + " (>10%). FLOAT=" +
                        Arrays.toString(sliceHead(refOut)) + " QUANT=" + Arrays.toString(sliceHead(quantOut)));
    }

    /**
     * Model-geometry variant: GQA 8:2, headDim 256, cache 26 with only the prefix filled —
     * exactly the failing pipeline's shapes. Catches GQA head-mapping or masked-tail bugs the
     * 1:1 small case cannot.
     */
    @Test
    public void testQuantisedDecodeAttnInlineScaleModelGeometry() {
        Nd4j.getRandom().setSeed(42);
        final int B = 1, qH = 8, kvH = 2, hd = 256, S = 26, filled = 16;
        final int cachePosVal = filled; // decode writes the current token at position 16
        final double scaleFactor = 1.0 / Math.sqrt(hd);

        INDArray query  = Nd4j.rand(DataType.FLOAT, B, 1, qH, hd).subi(0.5).muli(4.0);
        INDArray keyCur = Nd4j.rand(DataType.FLOAT, B, 1, kvH, hd).subi(0.5).muli(4.0);
        INDArray valCur = Nd4j.rand(DataType.FLOAT, B, 1, kvH, hd).subi(0.5).muli(4.0);
        // Cache: prefix [0,filled) random (prefill), rest zeros (padding) — pipeline layout.
        INDArray keyCacheF = Nd4j.zeros(DataType.FLOAT, B, S, kvH, hd);
        INDArray valCacheF = Nd4j.zeros(DataType.FLOAT, B, S, kvH, hd);
        keyCacheF.get(org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, filled),
                org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all())
                .assign(Nd4j.rand(DataType.FLOAT, B, filled, kvH, hd).subi(0.5).muli(4.0));
        valCacheF.get(org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, filled),
                org.nd4j.linalg.indexing.NDArrayIndex.all(), org.nd4j.linalg.indexing.NDArrayIndex.all())
                .assign(Nd4j.rand(DataType.FLOAT, B, filled, kvH, hd).subi(0.5).muli(4.0));
        INDArray cachePos = Nd4j.scalar((long) cachePosVal).castTo(DataType.INT64);
        // Decode mask [1,1,1,S]: positions [0,filled] visible (0), tail masked (-1e9).
        INDArray mask = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, S);
        for (int p = filled + 1; p < S; p++) mask.putScalar(0, 0, 0, p, -1e9f);

        INDArray refOut = execDpaV2WithBias(query, valCur, keyCur, keyCacheF.dup(), valCacheF.dup(),
                cachePos, scaleFactor, mask);
        INDArray keyCacheI8 = quantizeCacheInline(keyCacheF, B, S, kvH, hd);
        INDArray valCacheI8 = quantizeCacheInline(valCacheF, B, S, kvH, hd);
        INDArray quantOut = execDpaV2WithBias(query, valCur, keyCur, keyCacheI8, valCacheI8,
                cachePos, scaleFactor, mask);

        INDArray diff = refOut.sub(quantOut);
        double maxAbs = Nd4j.getExecutioner().execAndReturn(
                new org.nd4j.linalg.api.ops.impl.reduce.same.AMax(diff)).getFinalResult().doubleValue();
        double refMax = refOut.amaxNumber().doubleValue();
        double rel = refMax > 1e-9 ? maxAbs / refMax : maxAbs;
        log.info("[ISOLATION-geom] maxAbsErr={} refMax={} relErr={}", maxAbs, refMax, rel);
        assertTrue(rel < 0.10,
                "Model-geometry (GQA 8:2 hd=256) inline quantised attn vs float relErr " + rel
                        + " (>10%). FLOAT=" + Arrays.toString(sliceHead(refOut))
                        + " QUANT=" + Arrays.toString(sliceHead(quantOut)));
    }

    private INDArray execDpaV2WithBias(INDArray query, INDArray values, INDArray keys,
                                       INDArray keyCache, INDArray valueCache, INDArray cachePos,
                                       double scaleFactor, INDArray bias) {
        INDArray empty = Nd4j.empty(DataType.FLOAT);
        DynamicCustomOp op = DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(query, values, keys, empty, empty, keyCache, valueCache, cachePos, bias)
                .addFloatingPointArguments(scaleFactor, 0.0)
                .addBooleanArguments(false, false, true)
                .build();
        return Nd4j.exec(op)[0];
    }

    // ROW-INLINE quantize: [B,S,kvH,hd] float → INT8 [B,S,kvH,hd+4], each row = hd values ++ that
    // row's float32 scale (inside the logical tensor). Mirrors GenerationPipeline STEP 2.
    private INDArray quantizeCacheInline(INDArray cacheF, int B, int S, int kvH, int hd) {
        INDArray rowInline = Nd4j.exec(new KVCacheQuantize(cacheF, KVCacheQuantize.FORMAT_INT8, true))[0];
        Nd4j.getExecutioner().commit();
        rowInline.syncToHost();
        return rowInline;
    }

    // Quantise a [B,S,kvH,hd] float cache into INT8 [B,S,kvH,hd] + scales [B,S,kvH].
    private INDArray[] quantizeCache(INDArray cacheF, int rows, int hd, int B, int S, int kvH) {
        INDArray flat = cacheF.reshape(rows, hd);
        KVCacheQuantize q = new KVCacheQuantize(flat, KVCacheQuantize.FORMAT_INT8);
        INDArray[] r = Nd4j.exec(q);
        INDArray i8   = r[0].reshape(B, S, kvH, hd);
        INDArray sc   = r[1].reshape(B, S, kvH);
        return new INDArray[]{i8, sc};
    }

    private INDArray execDpaV2(INDArray query, INDArray values, INDArray keys,
                               INDArray keyCache, INDArray valueCache, INDArray cachePos,
                               double scaleFactor, INDArray keyScale, INDArray valScale) {
        INDArray empty = Nd4j.empty(DataType.FLOAT);
        DynamicCustomOp.DynamicCustomOpsBuilder b = DynamicCustomOp.builder("dot_product_attention_v2");
        if (keyScale != null) {
            b.addInputs(query, values, keys, empty, empty, keyCache, valueCache, cachePos, empty, keyScale, valScale);
        } else {
            b.addInputs(query, values, keys, empty, empty, keyCache, valueCache, cachePos, empty);
        }
        b.addFloatingPointArguments(scaleFactor, 0.0);
        b.addBooleanArguments(false, false, true); // useCausalMask, training, useFlashAttention
        DynamicCustomOp op = b.build();
        return Nd4j.exec(op)[0];
    }

    private float[] sliceHead(INDArray out) {
        // out is [B,1,qH,hd] — return the first head's headDim row.
        INDArray flat = out.reshape(out.length());
        int n = (int) Math.min(8, out.length());
        float[] r = new float[n];
        for (int i = 0; i < n; i++) r[i] = flat.getFloat(i);
        return r;
    }
}
