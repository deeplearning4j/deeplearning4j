package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests memory behavior during prefill→recompile→decode transition.
 * Simulates what happens in the VLM pipeline when a 679-token prefill
 * runs, then the plan is recompiled for seqLen=1 decode.
 *
 * Run: cd platform-tests && mvn test -Dtest=TestDSPPrefillRecompileMemory \
 *   -Dbackend.artifactId=nd4j-cuda-12.9 -Dlibnd4j.triton=ON 2>&1 | tee /tmp/prefill-recompile.log
 */
@Slf4j
public class TestDSPPrefillRecompileMemory {

    @Test
    public void testPrefillRecompileDoesNotLeakMemory() {
        int hidden = 576;
        int prefillLen = 679;
        int decodeLen = 1;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, -1, hidden);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.01));
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, hidden, 100).muli(0.01));

        SDVariable flat = sd.reshape(x, -1, hidden);
        SDVariable h = sd.mmul("h", flat, w1);
        SDVariable act = sd.nn.relu("act", h, 0);
        SDVariable out = sd.mmul("out", act, w2);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // === Phase 1: Prefill (large sequence) ===
        Nd4j.getExecutioner().commit();
        long freeBeforePrefill = Nd4j.getNativeOps().getDeviceFreeMemory(0);
        log.info("Before prefill: free={} MB", freeBeforePrefill / (1024 * 1024));

        INDArray prefillInput = Nd4j.randn(DataType.FLOAT, 1, prefillLen, hidden);
        sd.outputDirect(Map.of("x", prefillInput), "out");
        prefillInput.close();

        Nd4j.getExecutioner().commit();
        long freeAfterPrefill = Nd4j.getNativeOps().getDeviceFreeMemory(0);
        long prefillUsed = freeBeforePrefill - freeAfterPrefill;
        log.info("After prefill: free={} MB, used={} MB",
                freeAfterPrefill / (1024 * 1024), prefillUsed / (1024 * 1024));

        // === Phase 2: Recompile (clear plan, clear caches) ===
        sd.clearDynamicShapePlanCache();
        sd.getOrCreateSession().clearAllCaches();

        Nd4j.getExecutioner().commit();
        System.gc();
        try { Thread.sleep(100); } catch (InterruptedException ignored) {}
        Nd4j.getExecutioner().commit();

        long freeAfterRecompile = Nd4j.getNativeOps().getDeviceFreeMemory(0);
        long recompileRecovered = freeAfterRecompile - freeAfterPrefill;
        long stillLeaked = freeBeforePrefill - freeAfterRecompile;
        log.info("After recompile+clear: free={} MB, recovered={} MB, still leaked={} MB",
                freeAfterRecompile / (1024 * 1024),
                recompileRecovered / (1024 * 1024),
                stillLeaked / (1024 * 1024));

        // === Phase 3: Decode (small sequence, 50 steps) ===
        for (int i = 0; i < 50; i++) {
            INDArray decodeInput = Nd4j.randn(DataType.FLOAT, 1, decodeLen, hidden);
            sd.outputDirect(Map.of("x", decodeInput), "out");
        }

        Nd4j.getExecutioner().commit();
        long freeAfterDecode = Nd4j.getNativeOps().getDeviceFreeMemory(0);
        long decodeGrowth = freeAfterRecompile - freeAfterDecode;
        log.info("After 50 decode steps: free={} MB, decode growth={} MB",
                freeAfterDecode / (1024 * 1024), decodeGrowth / (1024 * 1024));

        // === Phase 4: Without clearAllCaches (compare) ===
        sd.close();

        SameDiff sd2 = SameDiff.create();
        SDVariable x2 = sd2.placeHolder("x", DataType.FLOAT, -1, -1, hidden);
        SDVariable w12 = sd2.constant("w1", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.01));
        SDVariable w22 = sd2.constant("w2", Nd4j.randn(DataType.FLOAT, hidden, 100).muli(0.01));
        SDVariable flat2 = sd2.reshape(x2, -1, hidden);
        SDVariable h2 = sd2.mmul("h", flat2, w12);
        SDVariable act2 = sd2.nn.relu("act", h2, 0);
        SDVariable out2 = sd2.mmul("out", act2, w22);

        sd2.setDspAutoCompileEnabled(true);
        sd2.setDspNativeAutoCompileEnabled(true);

        Nd4j.getExecutioner().commit();
        long freeBeforeNoClear = Nd4j.getNativeOps().getDeviceFreeMemory(0);
        log.info("Before no-clear test: free={} MB", freeBeforeNoClear / (1024 * 1024));

        INDArray prefill2 = Nd4j.randn(DataType.FLOAT, 1, prefillLen, hidden);
        sd2.outputDirect(Map.of("x", prefill2), "out");
        prefill2.close();

        // Recompile WITHOUT clearAllCaches
        sd2.clearDynamicShapePlanCache();
        // No clearAllCaches here

        Nd4j.getExecutioner().commit();
        System.gc();
        try { Thread.sleep(100); } catch (InterruptedException ignored) {}
        Nd4j.getExecutioner().commit();

        long freeAfterNoClear = Nd4j.getNativeOps().getDeviceFreeMemory(0);
        long noClearLeaked = freeBeforeNoClear - freeAfterNoClear;
        log.info("After recompile WITHOUT clear: free={} MB, leaked={} MB",
                freeAfterNoClear / (1024 * 1024), noClearLeaked / (1024 * 1024));

        for (int i = 0; i < 50; i++) {
            INDArray d = Nd4j.randn(DataType.FLOAT, 1, decodeLen, hidden);
            sd2.outputDirect(Map.of("x", d), "out");
        }

        Nd4j.getExecutioner().commit();
        long freeAfterNoClearDecode = Nd4j.getNativeOps().getDeviceFreeMemory(0);
        long noClearDecodeGrowth = freeAfterNoClear - freeAfterNoClearDecode;
        log.info("After 50 decode (no-clear): free={} MB, growth={} MB",
                freeAfterNoClearDecode / (1024 * 1024), noClearDecodeGrowth / (1024 * 1024));

        log.info("=== SUMMARY ===");
        log.info("  With clearAllCaches: prefill leaked {} MB, decode growth {} MB",
                stillLeaked / (1024 * 1024), decodeGrowth / (1024 * 1024));
        log.info("  Without clearAllCaches: prefill leaked {} MB, decode growth {} MB",
                noClearLeaked / (1024 * 1024), noClearDecodeGrowth / (1024 * 1024));

        sd2.close();
    }
}
