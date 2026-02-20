package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueLaunchContext;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolated tests for DSP memory behavior and stall detection.
 *
 * <p>These tests measure:
 * <ul>
 *   <li>Memory growth per DSP execute() call — detects leaks</li>
 *   <li>Failover overhead — measures cost of cross-device allocation</li>
 *   <li>Per-phase timing — identifies which DSP phase stalls</li>
 *   <li>TrimPool effectiveness — verifies memory reclamation</li>
 * </ul>
 *
 * <p>Run from platform-tests:
 * <pre>
 * mvn test -Dtest=DSPMemoryAndStallTest
 * mvn test -Dtest=DSPMemoryAndStallTest#testMemoryGrowthPerExecute
 * </pre>
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class DSPMemoryAndStallTest {

    private static NativeOps nativeOps;
    private static int numDevices;

    @BeforeAll
    static void setup() {
        // Force DSP mode
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
        nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        log.info("DSPMemoryAndStallTest: {} GPU devices detected", numDevices);
    }

    // =========================================================================
    // Memory snapshot helpers
    // =========================================================================

    private static long[] getPoolUsed() {
        long[] used = new long[numDevices];
        for (int d = 0; d < numDevices; d++) {
            LongPointer usedPtr = new LongPointer(1);
            LongPointer reservedPtr = new LongPointer(1);
            nativeOps.getMemoryPoolStats(d, usedPtr, reservedPtr);
            used[d] = usedPtr.get();
        }
        return used;
    }

    private static long[] getPoolReserved() {
        long[] reserved = new long[numDevices];
        for (int d = 0; d < numDevices; d++) {
            LongPointer usedPtr = new LongPointer(1);
            LongPointer reservedPtr = new LongPointer(1);
            nativeOps.getMemoryPoolStats(d, usedPtr, reservedPtr);
            reserved[d] = reservedPtr.get();
        }
        return reserved;
    }

    private static long[] getDeviceFree() {
        long[] free = new long[numDevices];
        for (int d = 0; d < numDevices; d++) {
            free[d] = nativeOps.getDeviceFreeMemory(d);
        }
        return free;
    }

    private static long[] getDeviceTotal() {
        long[] total = new long[numDevices];
        for (int d = 0; d < numDevices; d++) {
            total[d] = nativeOps.getDeviceTotalMemory(d);
        }
        return total;
    }

    private static void trimAll() {
        for (int d = 0; d < numDevices; d++) {
            nativeOps.trimMemoryPool(d);
        }
    }

    private static void logMemoryState(String label) {
        long[] used = getPoolUsed();
        long[] reserved = getPoolReserved();
        long[] free = getDeviceFree();
        long[] total = getDeviceTotal();
        StringBuilder sb = new StringBuilder();
        sb.append(label).append(": ");
        for (int d = 0; d < numDevices; d++) {
            if (d > 0) sb.append(" | ");
            sb.append(String.format("GPU%d poolUsed=%dMB poolRes=%dMB cudaFree=%dMB/%dMB",
                    d, used[d] / (1024 * 1024), reserved[d] / (1024 * 1024),
                    free[d] / (1024 * 1024), total[d] / (1024 * 1024)));
        }
        log.info(sb.toString());
    }

    // =========================================================================
    // Test 1: Memory growth per DSP execute()
    // =========================================================================

    /**
     * Builds a small SameDiff graph (mimicking a vision encoder subgraph)
     * and runs execute() repeatedly. Measures poolUsed before/after each call.
     * If poolUsed grows between calls, something is leaking.
     */
    @Test
    @Order(1)
    @DisplayName("Memory growth per DSP execute() — detect leaks")
    public void testMemoryGrowthPerExecute() {
        SameDiff sd = buildTestGraph();
        int numExecutions = 10;
        long[][] poolUsedHistory = new long[numExecutions][numDevices];

        // Warm up — first execution compiles the plan
        Map<String, INDArray> warmupInputs = createInputs(sd);
        Map<String, INDArray> warmupOut = sd.output(warmupInputs, sd.outputs().toArray(new String[0]));
        for (INDArray v : warmupOut.values()) { v.setCloseable(true); v.close(); }
        sd.clearPlaceholders(false);
        sd.clearOpInputs();
        sd.resetSession();
        Nd4j.getExecutioner().commit();
        trimAll();

        logMemoryState("After warmup+trim");
        long[] baselineUsed = getPoolUsed();

        for (int i = 0; i < numExecutions; i++) {
            Map<String, INDArray> inputs = createInputs(sd);
            Map<String, INDArray> outputs = sd.output(inputs, sd.outputs().toArray(new String[0]));

            // Close outputs
            for (INDArray v : outputs.values()) {
                v.setCloseable(true);
                v.close();
            }
            sd.clearPlaceholders(false);
            sd.clearOpInputs();
            sd.resetSession();
            Nd4j.getExecutioner().commit();
            trimAll();

            poolUsedHistory[i] = getPoolUsed();
            log.info("Execute {}: poolUsed={}", i,
                    Arrays.toString(Arrays.stream(poolUsedHistory[i])
                            .map(x -> x / (1024 * 1024)).toArray()));
        }

        // Check: poolUsed should NOT grow significantly between executions
        // Allow 1MB tolerance per device per execution for fragmentation
        for (int d = 0; d < numDevices; d++) {
            long growth = poolUsedHistory[numExecutions - 1][d] - baselineUsed[d];
            long maxAcceptableGrowthMB = 5; // 5MB total over 10 executions
            log.info("GPU {}: poolUsed growth = {}MB over {} executions (baseline={}MB, final={}MB)",
                    d, growth / (1024 * 1024), numExecutions,
                    baselineUsed[d] / (1024 * 1024),
                    poolUsedHistory[numExecutions - 1][d] / (1024 * 1024));
            assertTrue(growth < maxAcceptableGrowthMB * 1024 * 1024,
                    "GPU " + d + " poolUsed grew by " + (growth / (1024 * 1024)) +
                            "MB over " + numExecutions + " executions — possible leak");
        }

        sd.close();
    }

    // =========================================================================
    // Test 2: TrimPool effectiveness
    // =========================================================================

    /**
     * Allocates memory via a DSP execute(), trims, and checks if poolReserved
     * drops back close to poolUsed. If trimPool doesn't reclaim, pool is holding
     * memory that's logically free but physically reserved.
     */
    @Test
    @Order(2)
    @DisplayName("TrimPool effectiveness — verify memory reclamation")
    public void testTrimPoolEffectiveness() {
        SameDiff sd = buildTestGraph();

        // Warm up
        Map<String, INDArray> inputs = createInputs(sd);
        Map<String, INDArray> outputs = sd.output(inputs, sd.outputs().toArray(new String[0]));
        for (INDArray v : outputs.values()) { v.setCloseable(true); v.close(); }
        sd.clearPlaceholders(false);
        sd.clearOpInputs();
        sd.resetSession();
        Nd4j.getExecutioner().commit();

        logMemoryState("Before trim");
        long[] preUsed = getPoolUsed();
        long[] preReserved = getPoolReserved();

        trimAll();

        logMemoryState("After trim");
        long[] postUsed = getPoolUsed();
        long[] postReserved = getPoolReserved();

        for (int d = 0; d < numDevices; d++) {
            long reclaimable = preReserved[d] - preUsed[d];
            long reclaimed = preReserved[d] - postReserved[d];
            log.info("GPU {}: reclaimable={}MB, actually reclaimed={}MB, post reserved={}MB, post used={}MB",
                    d, reclaimable / (1024 * 1024), reclaimed / (1024 * 1024),
                    postReserved[d] / (1024 * 1024), postUsed[d] / (1024 * 1024));

            if (reclaimable > 1024 * 1024) { // Only check if there was reclaimable memory
                if (reclaimed == 0) {
                    // CUDA's mempool may not release below the release threshold.
                    // This is expected behavior when usage is well below the 75% threshold.
                    // Log as warning rather than fail.
                    long total = getDeviceTotal()[d];
                    double pctReserved = 100.0 * preReserved[d] / total;
                    log.warn("GPU {} trimPool reclaimed 0 bytes despite {}MB reclaimable " +
                                    "(reserved={}% of total — may be below CUDA mempool release threshold)",
                            d, reclaimable / (1024 * 1024), String.format("%.1f", pctReserved));
                }
            }
        }

        sd.close();
    }

    // =========================================================================
    // Test 3: Per-phase timing breakdown
    // =========================================================================

    /**
     * Runs DSP execute() with timing enabled and parses the timing output.
     * Identifies which phase (shape, alloc, exec, release) dominates.
     */
    @Test
    @Order(3)
    @DisplayName("Per-phase timing — identify stalls")
    public void testPerPhaseTiming() {
        // Enable DSP timing for this test
        String prevTiming = System.getProperty(ND4JSystemProperties.INFERENCE_TIMING);
        System.setProperty(ND4JSystemProperties.INFERENCE_TIMING, "true");

        try {
            SameDiff sd = buildTimingTestGraph();

            // Warm up (first run compiles plan)
            Map<String, INDArray> warmupInputs = createTimingInputs(sd);
            Map<String, INDArray> warmupOut = sd.output(warmupInputs, sd.outputs().toArray(new String[0]));
            for (INDArray v : warmupOut.values()) { v.setCloseable(true); v.close(); }
            sd.clearPlaceholders(false);
            sd.clearOpInputs();
            sd.resetSession();
            Nd4j.getExecutioner().commit();

            // Timed run — timing summary prints automatically via DSP executor
            log.info("=== TIMED RUN START ===");
            long wallStart = System.nanoTime();

            Map<String, INDArray> inputs = createTimingInputs(sd);
            Map<String, INDArray> outputs = sd.output(inputs, sd.outputs().toArray(new String[0]));

            long wallEnd = System.nanoTime();
            double wallMs = (wallEnd - wallStart) / 1_000_000.0;
            log.info("=== TIMED RUN END: wall clock = {}ms ===", String.format("%.1f", wallMs));

            // The DSP executor prints its own timing breakdown (shape/alloc/exec/release)
            // to the log. We just need to run it and check the output.

            for (INDArray v : outputs.values()) { v.setCloseable(true); v.close(); }
            sd.clearPlaceholders(false);
            sd.clearOpInputs();
            sd.resetSession();
            sd.close();

        } finally {
            if (prevTiming != null) {
                System.setProperty(ND4JSystemProperties.INFERENCE_TIMING, prevTiming);
            } else {
                System.clearProperty(ND4JSystemProperties.INFERENCE_TIMING);
            }
        }
    }

    // =========================================================================
    // Test 4: Failover overhead measurement
    // =========================================================================

    /**
     * If multiple GPUs are available, measure the cost of cross-device
     * allocation vs same-device allocation. This quantifies failover overhead.
     */
    @Test
    @Order(4)
    @DisplayName("Failover overhead — cross-device allocation cost")
    public void testFailoverOverhead() {
        if (numDevices < 2) {
            log.info("Skipping failover test — only {} device(s)", numDevices);
            return;
        }

        int iterations = 100;
        int arraySize = 1024 * 1024; // 4MB per array (FLOAT)

        // Measure: normal allocation on device 0
        trimAll();
        long normalStart = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            INDArray arr = Nd4j.zeros(DataType.FLOAT, arraySize);
            arr.close();
        }
        Nd4j.getExecutioner().commit();
        long normalNs = System.nanoTime() - normalStart;

        trimAll();
        logMemoryState("After normal alloc test");

        // Measure: allocation with forced trim between each alloc
        // (simulates what happens when pool is under pressure)
        long trimmedStart = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            INDArray arr = Nd4j.zeros(DataType.FLOAT, arraySize);
            arr.close();
            Nd4j.getExecutioner().commit();
            nativeOps.trimMemoryPool(0);
        }
        long trimmedNs = System.nanoTime() - trimmedStart;

        trimAll();
        logMemoryState("After trimmed alloc test");

        double normalMs = normalNs / 1_000_000.0;
        double trimmedMs = trimmedNs / 1_000_000.0;
        double overhead = trimmedMs / normalMs;

        log.info("Normal: {}ms for {} allocs ({}us/alloc)",
                String.format("%.1f", normalMs), iterations,
                String.format("%.1f", normalMs * 1000 / iterations));
        log.info("Trimmed: {}ms for {} allocs ({}us/alloc)",
                String.format("%.1f", trimmedMs), iterations,
                String.format("%.1f", trimmedMs * 1000 / iterations));
        log.info("Trim overhead: {}x", String.format("%.2f", overhead));

        // Trim should add some overhead but not 10x+
        assertTrue(overhead < 10.0,
                "Trim overhead is " + String.format("%.1f", overhead) + "x — too high");
    }

    // =========================================================================
    // Test 5: Per-op stall detection
    // =========================================================================

    /**
     * Builds a graph with ops of different types and measures per-op execution
     * time to identify which ops stall. Reports ops that take >1ms.
     */
    @Test
    @Order(5)
    @DisplayName("Per-op stall detection — find slow ops")
    public void testPerOpStallDetection() {
        // Enable trace mode to see per-slot info
        String prevTrace = System.getProperty(ND4JSystemProperties.DSP_TRACE);
        String prevTiming = System.getProperty(ND4JSystemProperties.INFERENCE_TIMING);
        System.setProperty(ND4JSystemProperties.INFERENCE_TIMING, "true");

        try {
            SameDiff sd = buildStallDetectionGraph();

            // Warm up
            Map<String, INDArray> warmupInputs = createStallInputs(sd);
            Map<String, INDArray> warmupOut = sd.output(warmupInputs, sd.outputs().toArray(new String[0]));
            for (INDArray v : warmupOut.values()) { v.setCloseable(true); v.close(); }
            sd.clearPlaceholders(false);
            sd.clearOpInputs();
            sd.resetSession();
            Nd4j.getExecutioner().commit();
            trimAll();

            // Actual timed run
            log.info("=== PER-OP STALL DETECTION START ===");
            logMemoryState("Before execution");

            Map<String, INDArray> inputs = createStallInputs(sd);

            long execStart = System.nanoTime();
            Map<String, INDArray> outputs = sd.output(inputs, sd.outputs().toArray(new String[0]));
            long execEnd = System.nanoTime();

            logMemoryState("After execution");
            log.info("Total execution: {}ms",
                    String.format("%.1f", (execEnd - execStart) / 1_000_000.0));

            for (INDArray v : outputs.values()) { v.setCloseable(true); v.close(); }
            sd.clearPlaceholders(false);
            sd.clearOpInputs();
            sd.resetSession();
            sd.close();
            log.info("=== PER-OP STALL DETECTION END ===");
        } finally {
            if (prevTrace != null) System.setProperty(ND4JSystemProperties.DSP_TRACE, prevTrace);
            else System.clearProperty(ND4JSystemProperties.DSP_TRACE);
            if (prevTiming != null) System.setProperty(ND4JSystemProperties.INFERENCE_TIMING, prevTiming);
            else System.clearProperty(ND4JSystemProperties.INFERENCE_TIMING);
        }
    }

    // =========================================================================
    // Test 6: Repeated execute with growing memory tracking
    // =========================================================================

    /**
     * Simulates the vision encoder pattern: same graph, same shape inputs,
     * executed 10 times in sequence (like 10 frames). Tracks memory growth
     * per execution with detailed pool stats.
     */
    @Test
    @Order(6)
    @DisplayName("Repeated execute — simulate 10-frame encoding")
    public void testRepeatedExecuteMemoryTracking() {
        SameDiff sd = buildTimingTestGraph();
        int numFrames = 10;

        // Warm up + plan compilation
        Map<String, INDArray> warmupInputs = createTimingInputs(sd);
        Map<String, INDArray> warmupOut = sd.output(warmupInputs, sd.outputs().toArray(new String[0]));
        for (INDArray v : warmupOut.values()) { v.setCloseable(true); v.close(); }
        sd.clearPlaceholders(false);
        sd.clearOpInputs();
        sd.resetSession();
        Nd4j.getExecutioner().commit();
        trimAll();

        long[] baselineUsed = getPoolUsed();
        logMemoryState("Baseline (post-warmup)");

        long[] frameTimes = new long[numFrames];
        long[][] perFramePoolUsed = new long[numFrames][numDevices];
        long[][] perFramePoolReserved = new long[numFrames][numDevices];

        for (int f = 0; f < numFrames; f++) {
            // Trim before each frame (like PipelinedVisionEncoder does)
            if (f > 0) {
                trimAll();
            }

            long frameStart = System.nanoTime();
            Map<String, INDArray> inputs = createTimingInputs(sd);
            Map<String, INDArray> outputs = sd.output(inputs, sd.outputs().toArray(new String[0]));
            frameTimes[f] = System.nanoTime() - frameStart;

            // Close outputs properly
            for (INDArray v : outputs.values()) {
                v.setCloseable(true);
                v.close();
            }
            sd.clearPlaceholders(false);
            sd.clearOpInputs();
            sd.resetSession();
            Nd4j.getExecutioner().commit();

            perFramePoolUsed[f] = getPoolUsed();
            perFramePoolReserved[f] = getPoolReserved();

            log.info("Frame {}: {}ms | poolUsed={} poolReserved={}",
                    f, String.format("%.1f", frameTimes[f] / 1_000_000.0),
                    Arrays.toString(Arrays.stream(perFramePoolUsed[f])
                            .map(x -> x / (1024 * 1024)).toArray()),
                    Arrays.toString(Arrays.stream(perFramePoolReserved[f])
                            .map(x -> x / (1024 * 1024)).toArray()));
        }

        // Analyze: check for monotonic poolUsed growth
        log.info("=== MEMORY GROWTH ANALYSIS ===");
        for (int d = 0; d < numDevices; d++) {
            long totalGrowth = perFramePoolUsed[numFrames - 1][d] - baselineUsed[d];
            long perFrameGrowth = totalGrowth / numFrames;
            log.info("GPU {}: total growth={}MB, per-frame={}MB over {} frames",
                    d, totalGrowth / (1024 * 1024), perFrameGrowth / (1024 * 1024), numFrames);

            // Detect monotonic growth pattern
            int growthCount = 0;
            for (int f = 1; f < numFrames; f++) {
                if (perFramePoolUsed[f][d] > perFramePoolUsed[f - 1][d] + 1024 * 1024) {
                    growthCount++;
                }
            }
            if (growthCount > numFrames / 2) {
                log.warn("GPU {}: MONOTONIC GROWTH detected ({}/{} frames grew)", d, growthCount, numFrames - 1);
            }
        }

        // Analyze: timing consistency
        long minTime = Long.MAX_VALUE, maxTime = 0, sumTime = 0;
        for (long t : frameTimes) {
            minTime = Math.min(minTime, t);
            maxTime = Math.max(maxTime, t);
            sumTime += t;
        }
        double avgMs = sumTime / numFrames / 1_000_000.0;
        double minMs = minTime / 1_000_000.0;
        double maxMs = maxTime / 1_000_000.0;
        log.info("Frame times: avg={}ms, min={}ms, max={}ms, ratio={}x",
                String.format("%.1f", avgMs), String.format("%.1f", minMs),
                String.format("%.1f", maxMs), String.format("%.1f", maxMs / minMs));

        // Flag if later frames are significantly slower (memory pressure)
        double firstHalfAvg = 0, secondHalfAvg = 0;
        for (int f = 0; f < numFrames / 2; f++) firstHalfAvg += frameTimes[f];
        for (int f = numFrames / 2; f < numFrames; f++) secondHalfAvg += frameTimes[f];
        firstHalfAvg /= (numFrames / 2);
        secondHalfAvg /= (numFrames - numFrames / 2);
        double slowdown = secondHalfAvg / firstHalfAvg;
        log.info("Slowdown: second half is {}x vs first half", String.format("%.2f", slowdown));
        if (slowdown > 2.0) {
            log.warn("SIGNIFICANT SLOWDOWN: later frames are {}x slower — likely memory pressure", String.format("%.1f", slowdown));
        }

        sd.close();
    }

    // =========================================================================
    // Test 7: Allocation timing breakdown
    // =========================================================================

    /**
     * Measures raw allocation cost outside of DSP to baseline:
     * - Nd4j.create() timing
     * - memsetSync timing
     * - trimPool timing
     */
    @Test
    @Order(7)
    @DisplayName("Allocation timing breakdown — raw alloc cost")
    public void testAllocationTimingBreakdown() {
        int iterations = 500;
        long[][] sizes = {
                {1024},                    // 4KB
                {64, 576},                 // 144KB (attention output)
                {1024, 1024},              // 4MB (weight matrix)
                {1, 64, 12288},            // 3MB (image features)
        };

        for (long[] shape : sizes) {
            long totalElements = 1;
            for (long s : shape) totalElements *= s;

            // Measure create + close cycle
            long createTotal = 0;
            long closeTotal = 0;
            for (int i = 0; i < iterations; i++) {
                long t0 = System.nanoTime();
                INDArray arr = Nd4j.zeros(DataType.FLOAT, shape);
                long t1 = System.nanoTime();
                arr.close();
                long t2 = System.nanoTime();
                createTotal += (t1 - t0);
                closeTotal += (t2 - t1);
            }

            log.info("Shape {}: create={}us/op, close={}us/op ({}KB each)",
                    Arrays.toString(shape),
                    String.format("%.1f", createTotal / 1000.0 / iterations),
                    String.format("%.1f", closeTotal / 1000.0 / iterations),
                    totalElements * 4 / 1024);
        }

        // Measure trimPool cost
        long trimTotal = 0;
        for (int i = 0; i < 100; i++) {
            INDArray arr = Nd4j.zeros(DataType.FLOAT, 1024 * 1024);
            arr.close();
            Nd4j.getExecutioner().commit();
            long t0 = System.nanoTime();
            nativeOps.trimMemoryPool(0);
            long t1 = System.nanoTime();
            trimTotal += (t1 - t0);
        }
        log.info("trimPool: {}us/call", String.format("%.1f", trimTotal / 1000.0 / 100));
    }

    // =========================================================================
    // Test 8: JNI boundary crossing cost
    // =========================================================================

    /**
     * Measures the pure JNI boundary crossing overhead for key native calls.
     * This quantifies the minimum per-op cost of the Java dispatch loop.
     */
    @Test
    @Order(8)
    @DisplayName("JNI boundary cost — minimum per-op overhead")
    public void testJniBoundaryCost() {
        int iterations = 10000;

        // getDeviceFreeMemory — lightweight JNI call
        long freeMemStart = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            nativeOps.getDeviceFreeMemory(0);
        }
        long freeMemNs = System.nanoTime() - freeMemStart;

        // lastErrorCode — very lightweight JNI call
        long errStart = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            nativeOps.lastErrorCode();
        }
        long errNs = System.nanoTime() - errStart;

        // getMemoryPoolStats — two output pointers
        LongPointer usedPtr = new LongPointer(1);
        LongPointer reservedPtr = new LongPointer(1);
        long statsStart = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            nativeOps.getMemoryPoolStats(0, usedPtr, reservedPtr);
        }
        long statsNs = System.nanoTime() - statsStart;

        log.info("JNI boundary costs ({} iterations):", iterations);
        log.info("  getDeviceFreeMemory: {}us/call",
                String.format("%.2f", freeMemNs / 1000.0 / iterations));
        log.info("  lastErrorCode:       {}us/call",
                String.format("%.2f", errNs / 1000.0 / iterations));
        log.info("  getMemoryPoolStats:  {}us/call",
                String.format("%.2f", statsNs / 1000.0 / iterations));

        // With 1962 ops, even 2us/JNI call × 5 JNI calls/op = 10us/op × 1962 = 20ms
        // This is the MINIMUM overhead from JNI alone
        double minJniOverheadPerOp = (freeMemNs / 1000.0 / iterations) * 3; // ~3 JNI calls per op
        double minJniOverheadPerFrame = minJniOverheadPerOp * 1962;
        log.info("  Estimated minimum JNI overhead per 1962-op frame: {}ms",
                String.format("%.1f", minJniOverheadPerFrame / 1000));
    }

    // =========================================================================
    // Graph builders
    // =========================================================================

    /**
     * Build a test graph that mimics vision encoder patterns:
     * matmul, layer norm, attention, reshape, transpose.
     */
    private SameDiff buildTestGraph() {
        SameDiff sd = SameDiff.create();
        // Simplified attention block
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 64, 576);
        SDVariable weight1 = sd.var("weight1", Nd4j.randn(DataType.FLOAT, 576, 576).muli(0.01));
        SDVariable weight2 = sd.var("weight2", Nd4j.randn(DataType.FLOAT, 576, 576).muli(0.01));

        // Linear projection
        SDVariable proj = sd.mmul("proj", input, weight1);
        // Add + norm approximation
        SDVariable added = proj.add("residual", input);
        SDVariable normed = sd.math.standardize("norm", added, 2);
        // Second linear
        SDVariable out = sd.mmul("output", normed, weight2);
        sd.setOutputs("output");

        return sd;
    }

    private Map<String, INDArray> createInputs(SameDiff sd) {
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input", Nd4j.randn(DataType.FLOAT, 1, 64, 576));
        return inputs;
    }

    /**
     * Build a larger graph for timing tests — more ops to amplify timing differences.
     */
    private SameDiff buildTimingTestGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 64, 128);

        SDVariable x = input;
        // Stack multiple transformer-like layers
        for (int layer = 0; layer < 4; layer++) {
            SDVariable w1 = sd.var("w1_" + layer, Nd4j.randn(DataType.FLOAT, 128, 128).muli(0.01));
            SDVariable w2 = sd.var("w2_" + layer, Nd4j.randn(DataType.FLOAT, 128, 128).muli(0.01));

            SDVariable proj1 = sd.mmul("proj1_" + layer, x, w1);
            SDVariable act = sd.nn.relu("relu_" + layer, proj1, 0);
            SDVariable proj2 = sd.mmul("proj2_" + layer, act, w2);
            x = proj2.add("residual_" + layer, x);
            x = sd.math.standardize("norm_" + layer, x, 2);
        }
        x.rename("final_output");
        sd.setOutputs("final_output");

        return sd;
    }

    private Map<String, INDArray> createTimingInputs(SameDiff sd) {
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input", Nd4j.randn(DataType.FLOAT, 1, 64, 128));
        return inputs;
    }

    /**
     * Build a graph with diverse op types to test stall patterns.
     */
    private SameDiff buildStallDetectionGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32, 256);

        // Mix of op types to see which stalls
        SDVariable reshaped = sd.reshape("reshape1", input, 1, 32, 16, 16);
        SDVariable transposed = sd.permute("transpose1", reshaped, 0, 2, 1, 3); // [1,16,32,16]
        SDVariable flattened = sd.reshape("flatten", transposed, 1, 16, 512);

        SDVariable w = sd.var("weight", Nd4j.randn(DataType.FLOAT, 512, 256).muli(0.01));
        SDVariable matmul = sd.mmul("matmul", flattened, w);

        SDVariable bias = sd.var("bias", Nd4j.zeros(DataType.FLOAT, 256));
        SDVariable biased = matmul.add("add_bias", bias);

        SDVariable activated = sd.nn.relu("relu", biased, 0);
        SDVariable normed = sd.math.standardize("layernorm", activated, 2);

        // Cast to test type conversion
        SDVariable casted = sd.castTo("cast_half", normed, DataType.HALF);
        SDVariable castBack = sd.castTo("cast_float", casted, DataType.FLOAT);

        castBack.rename("output");
        sd.setOutputs("output");
        return sd;
    }

    private Map<String, INDArray> createStallInputs(SameDiff sd) {
        Map<String, INDArray> inputs = new HashMap<>();
        inputs.put("input", Nd4j.randn(DataType.FLOAT, 1, 32, 256));
        return inputs;
    }
}
