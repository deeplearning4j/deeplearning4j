/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * Focused CPU benchmark for the bge-base-en-v1.5 (BERT-base) embedding model at the
 * crawl's fixed max shape [32 x 512]. Measures the effect of OpenBLAS thread count on
 * one warmed DSP plan by A/B-ing 1 thread vs all cores, then reports a logical-batch
 * latency table. NOT a CI gate — skips gracefully if the model is absent.
 *
 * Run (CPU backend):
 *   cd platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *     -Dbackend.artifactId=nd4j-native \
 *     -Dtest=BgeCpuThreadingBenchmark 2>&1 | tee /tmp/bge-bench.log
 */
public class BgeCpuThreadingBenchmark {

    private static final Logger log = LoggerFactory.getLogger(BgeCpuThreadingBenchmark.class);

    // Batch is configurable (-Dbge.batch=N). The full crawl shape is 32, but the [32x512]
    // BGE plan needs ~44GB of *native* (non-javacpp-capped) DSP intermediates; default to 16
    // (~22GB) so this runs alongside other memory consumers without OOM. The 1-vs-N-thread
    // speedup ratio is essentially batch-independent.
    private static final int BATCH = Integer.getInteger("bge.batch", 16);
    // Seq is configurable (-Dbge.seq=N). Attention scratch is O(seq^2), so seq dominates the
    // ~45GB footprint of the unfused [*,512] plan; drop it to fit under the memory cap while
    // profiling. Default 512 (the real crawl shape).
    private static final int SEQ = Integer.getInteger("bge.seq", 512);

    @Test
    public void benchmark32x512() throws Exception {
        File modelFile = new File(System.getProperty("bge.model.path",
                "/home/agibsonccc/.kompile/models/bge-base-en-v1.5/model.opt.sdz"));
        Assumptions.assumeTrue(modelFile.exists(),
                "Pre-optimized model not found (run #saveOptimizedModel first): " + modelFile.getAbsolutePath());

        // JUST load the pre-optimized graph — no in-memory optimize, no raw graph + dup() held.
        SameDiff model = SDZSerializer.load(modelFile, true);
        log.info("BGE loaded (pre-optimized). ops={} inputs={} outputs={}",
                model.ops().length, model.inputs(), model.outputs());

        NativeOps ops = Nd4j.getNativeOps();
        int cores = Runtime.getRuntime().availableProcessors();
        log.info("cores={} ompGetMaxThreads={} openBlasThreads(current)={}",
                cores, ops.ompGetMaxThreads(), ops.getOpenBlasThreads());

        // Warm up the single fixed [32 x 512] DSP plan (default thread setting from the build).
        log.info("Warming [{} x {}] plan (3 passes)...", BATCH, SEQ);
        for (int w = 0; w < 3; w++) runOnce(model);
        log.info("Warm-up complete. RSS={} MB", rssMb());

        // A/B: 1 thread (old pinned behavior) vs all cores (the fix). oneDNN is ON for both,
        // so this isolates the BLAS-GEMM threading contribution.
        double median1 = timeSetting(model, ops, 1, 3);
        double medianN = timeSetting(model, ops, cores, 5);

        double speedup = median1 / medianN;
        log.info("================ BGE [32 x 512] CPU RESULT ================");
        log.info("  openBLAS=1     : {} ms / [32x512] pass (median)", fmt(median1));
        log.info("  openBLAS={} : {} ms / [32x512] pass (median)", cores, fmt(medianN));
        log.info("  BLAS threading speedup : {}x", fmt(speedup));
        log.info("  (oneDNN kernels active in both; prior no-oneDNN 1-thread build was ~68000-82000 ms/pass)");

        // Logical-batch latency table at the multi-threaded (shipped) setting. The crawl pads
        // every logical batch up to [32 x 512], so N texts cost ceil(N/32) full passes.
        log.info("  --- logical-batch latency @ {} threads (one warmed [32x512] plan) ---", cores);
        for (int logical : new int[]{1, 3, 12, 32, 64}) {
            int passes = (int) Math.ceil(logical / (double) BATCH);
            double totalMs = passes * medianN;
            log.info("    {} text(s): {} pass(es) => {} ms total, {} ms/text",
                    logical, passes, fmt(totalMs), fmt(totalMs / logical));
        }
        log.info("  RSS(final)={} MB", rssMb());
        log.info("===========================================================");

        // Restore the shipped default so we don't leave the JVM pinned.
        ops.setOpenBlasThreads(cores);
    }

    /** Profile WHERE the per-pass time goes: enable native op-timing, run N passes, print top ops. */
    @Test
    public void opTimingProfile() throws Exception {
        File modelFile = new File(System.getProperty("bge.model.path",
                "/home/agibsonccc/.kompile/models/bge-base-en-v1.5/model.opt.sdz"));
        Assumptions.assumeTrue(modelFile.exists(),
                "Pre-optimized model not found (run #saveOptimizedModel first): " + modelFile.getAbsolutePath());
        SameDiff model = SDZSerializer.load(modelFile, true);
        log.info("op-timing profile: ops={} shape=[{} x {}]", model.ops().length, BATCH, SEQ);

        NativeOps ops = Nd4j.getNativeOps();
        for (int w = 0; w < 4; w++) runOnce(model);   // warm + freeze the plan

        // A/B OMP threads (this drives oneDNN's threading, NOT the OpenBLAS knob) to see whether
        // the ~22ms oneDNN math ops are single-threaded-compute-bound or per-call-overhead-bound.
        int cores = Runtime.getRuntime().availableProcessors();
        for (int t : new int[]{1, cores}) {
            ops.setOmpNumThreads(t);
            runOnce(model);                           // warm at this thread count
            ops.resetOpTiming();
            ops.setOpTimingEnabled(1, 0);
            int N = 5;
            for (int i = 0; i < N; i++) runOnce(model);
            ops.flushOpTiming();
            log.info("================ OP TIMING @ OMP threads={} ({} passes @ [{} x {}]) ================",
                    t, N, BATCH, SEQ);
            ops.printOpTimingStats(15);
            ops.setOpTimingEnabled(0, 0);
        }
        ops.setOmpNumThreads(cores);
    }

    /** One-time: optimize model.sdz and SAVE it as model.opt.sdz, so runtime JUST loads it. */
    @Test
    public void saveOptimizedModel() throws Exception {
        File rawFile = new File(System.getProperty("bge.raw.model.path",
                "/home/agibsonccc/.kompile/models/bge-base-en-v1.5/model.sdz"));
        Assumptions.assumeTrue(rawFile.exists(), "raw model not found: " + rawFile.getAbsolutePath());
        File optFile = new File(rawFile.getParentFile(), "model.opt.sdz");

        SameDiff raw = SDZSerializer.load(rawFile, true);
        log.info("Optimizing+saving: {} ({} ops) -> {}", rawFile.getName(), raw.ops().length, optFile.getName());
        long t0 = System.currentTimeMillis();
        SDZSerializer.saveOptimized(raw, optFile, false, null, raw.outputs());
        log.info(">>> saveOptimized done in {} ms. {} MB -> {} MB",
                System.currentTimeMillis() - t0, rawFile.length() / 1048576, optFile.length() / 1048576);

        // Verify by loading just the optimized file (fresh graph).
        SameDiff opt = SDZSerializer.load(optFile, true);
        dumpGraph(opt, "SAVED+RELOADED " + optFile.getName());
    }

    /** Load the crawl's graph (model.sdz), run GraphOptimizer, show before/after — NO inference. */
    @Test
    public void printSummary() throws Exception {
        File modelFile = new File(System.getProperty("bge.model.path",
                "/home/agibsonccc/.kompile/models/bge-base-en-v1.5/model.sdz"));
        Assumptions.assumeTrue(modelFile.exists(), "BGE model not found at: " + modelFile.getAbsolutePath());

        SameDiff model = SDZSerializer.load(modelFile, true);
        dumpGraph(model, "RAW (" + modelFile.getName() + ")");

        long t0 = System.currentTimeMillis();
        SameDiff opt = org.nd4j.autodiff.samediff.optimize.GraphOptimizer.optimize(model, model.outputs());
        log.info(">>> GraphOptimizer.optimize took {} ms: {} ops -> {} ops (removed {})",
                System.currentTimeMillis() - t0, model.ops().length, opt.ops().length,
                model.ops().length - opt.ops().length);
        dumpGraph(opt, "OPTIMIZED");
    }

    /** Log op count, top op-type histogram, and an attention-fusion / shape-plumbing verdict. */
    private void dumpGraph(SameDiff model, String label) {
        org.nd4j.autodiff.functions.DifferentialFunction[] ops = model.ops();
        Map<String, Integer> hist = new HashMap<>();
        for (org.nd4j.autodiff.functions.DifferentialFunction op : ops) hist.merge(op.opName(), 1, Integer::sum);
        int fused = hist.entrySet().stream().filter(e -> e.getKey().toLowerCase().contains("attention"))
                .mapToInt(Map.Entry::getValue).sum();
        int mm = hist.getOrDefault("matmul", 0) + hist.getOrDefault("mmul", 0) + hist.getOrDefault("tensordot", 0);
        int sm = hist.getOrDefault("softmax", 0);
        int shape = hist.getOrDefault("shape_of", 0) + hist.getOrDefault("gather", 0) + hist.getOrDefault("concat", 0)
                + hist.getOrDefault("strided_slice", 0) + hist.getOrDefault("stack", 0) + hist.getOrDefault("size_at", 0)
                + hist.getOrDefault("reshape", 0) + hist.getOrDefault("rank", 0) + hist.getOrDefault("expand_dims", 0);
        log.info("================ {} : inputs={} outputs={} ops={} ================",
                label, model.inputs(), model.outputs(), ops.length);
        hist.entrySet().stream().sorted((a, b) -> b.getValue() - a.getValue()).limit(15)
                .forEach(e -> log.info("  {} x {}", e.getValue(), e.getKey()));
        log.info("  --> attention-fusion={}  matmul/tensordot={}  softmax={}  shape-plumbing={}  [{}]",
                fused, mm, sm, shape, fused > 0 ? "FUSED" : "UNFUSED");
    }

    /** Set OpenBLAS threads, warm once, then time {@code iters} passes; returns median ms. */
    private double timeSetting(SameDiff model, NativeOps ops, int threads, int iters) {
        ops.setOpenBlasThreads(threads);
        log.info("--- timing openBlasThreads set to {} (effective={}) ---", threads, ops.getOpenBlasThreads());
        runOnce(model); // warm at this setting
        double[] times = new double[iters];
        for (int i = 0; i < iters; i++) {
            Map<String, INDArray> in = makeInputs(model);
            long t0 = System.nanoTime();
            Map<String, INDArray> out = model.output(in, model.outputs());
            double ms = (System.nanoTime() - t0) / 1e6;
            times[i] = ms;
            for (INDArray o : out.values()) {
                assertFalse(o.isNaN().any(), "output contains NaN at threads=" + threads);
            }
            log.info("    [{}t] iter {}/{}: {} ms", threads, i + 1, iters, fmt(ms));
            closeAll(out);
            closeAll(in);
        }
        return median(times);
    }

    private void runOnce(SameDiff model) {
        Map<String, INDArray> in = makeInputs(model);
        Map<String, INDArray> out = model.output(in, model.outputs());
        closeAll(out);
        closeAll(in);
    }

    /** [32 x 512] INT64 inputs: CLS/SEP at positions 0/1 of every row, mask 1 on those two. */
    private Map<String, INDArray> makeInputs(SameDiff model) {
        Map<String, INDArray> m = new HashMap<>();
        for (String name : model.inputs()) {
            INDArray in = Nd4j.zeros(DataType.INT64, BATCH, SEQ);
            String ln = name.toLowerCase();
            if (ln.contains("input_id")) {
                for (int r = 0; r < BATCH; r++) { in.putScalar(r, 0, 101); in.putScalar(r, 1, 102); }
            } else if (ln.contains("attention") || ln.contains("mask")) {
                for (int r = 0; r < BATCH; r++) { in.putScalar(r, 0, 1); in.putScalar(r, 1, 1); }
            }
            m.put(name, in);
        }
        return m;
    }

    private static void closeAll(Map<String, INDArray> arrays) {
        for (INDArray a : arrays.values()) {
            if (a != null && !a.wasClosed()) a.close();
        }
    }

    private static double median(double[] xs) {
        double[] c = xs.clone();
        java.util.Arrays.sort(c);
        int n = c.length;
        return (n % 2 == 1) ? c[n / 2] : (c[n / 2 - 1] + c[n / 2]) / 2.0;
    }

    private static long rssMb() {
        try {
            for (String line : java.nio.file.Files.readAllLines(java.nio.file.Paths.get("/proc/self/status"))) {
                if (line.startsWith("VmRSS:")) {
                    return Long.parseLong(line.replaceAll("[^0-9]", "")) / 1024;
                }
            }
        } catch (Exception ignored) { }
        return -1;
    }

    private static String fmt(double v) {
        return String.format("%.1f", v);
    }
}
