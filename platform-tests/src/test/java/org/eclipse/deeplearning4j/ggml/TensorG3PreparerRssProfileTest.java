package org.eclipse.deeplearning4j.ggml;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryUsage;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Desktop reproduction of the on-device calibration RSS death: runs the FULL
 * SdxGgufModelPreparer.prepare() flow (streamed Q4 derivative -> canonical SDZ ->
 * calibration -> compiled assets) on the real Qwen3.5-0.8B BF16 GGUF with staged tokenizers,
 * sampling VmRSS every second and logging peak/final.
 */
@Slf4j
public class TensorG3PreparerRssProfileTest {

    private static final String MODEL_PATH =
            System.getProperty("tensor.g3.preparer.model",
                    System.getProperty("user.home")
                            + "/.cache/dl4j-llm-models/Qwen3.5-0.8B-BF16.gguf");
    private static final String TOKENIZER_DIR =
            System.getProperty("tensor.g3.preparer.tokenizer",
                    System.getProperty("user.home")
                            + "/.cache/dl4j-llm-models/Qwen3.5-0.8B-serving");
    private static final String CACHE_DIR =
            System.getProperty("tensor.g3.preparer.cache", "/tmp/tensor-g3-qual-cache");

    private static long rssMb() {
        try {
            for (String line : Files.readAllLines(Paths.get("/proc/self/status"))) {
                if (line.startsWith("VmRSS:")) {
                    return Long.parseLong(line.replaceAll("\\D+", "")) / 1024;
                }
            }
        } catch (Exception ignored) {
        }
        return -1;
    }

    private static long megabytes(long bytes) {
        return bytes / (1024 * 1024);
    }

    private static void logMemory(String phase) {
        MemoryUsage heap = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage();
        NativeOps nativeOps = Nd4j.getNativeOps();
        log.info("PREP_MEMORY phase={} rss_mb={} heap_used_mb={} heap_committed_mb={} "
                        + "javacpp_total_mb={} javacpp_physical_mb={} deallocator_refs={} "
                        + "shape_cache_mb={} tad_cache_mb={}",
                phase,
                rssMb(),
                megabytes(heap.getUsed()),
                megabytes(heap.getCommitted()),
                megabytes(Pointer.totalBytes()),
                megabytes(Pointer.physicalBytes()),
                Nd4j.getDeallocatorService().getReferenceMap().size(),
                megabytes(nativeOps.getShapeCachedBytes()),
                megabytes(nativeOps.getTADCachedBytes()));
    }

    @Test
    public void profileFullPreparation() throws Exception {
        Files.createDirectories(Paths.get(CACHE_DIR));
        log.info("PREP_RSS baseline_mb={}", rssMb());

        final AtomicBoolean running = new AtomicBoolean(true);
        final AtomicLong peak = new AtomicLong(rssMb());
        Thread sampler = new Thread(() -> {
            while (running.get()) {
                long r = rssMb();
                if (r > peak.get()) {
                    peak.set(r);
                }
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    return;
                }
            }
        }, "prep-rss-sampler");
        sampler.setDaemon(true);
        sampler.start();

        try {
            // Reflect into the package-private preparer: same entry the C API uses.
            Class<?> cls = Class.forName("org.eclipse.deeplearning4j.sdx.aot.SdxGgufModelPreparer");
            Method prepare = cls.getDeclaredMethod("prepare",
                    String.class, String.class, String.class, String.class, String.class);
            prepare.setAccessible(true);
            Object result = prepare.invoke(null,
                    MODEL_PATH,
                    TOKENIZER_DIR,
                    "android-arm64-nnapi-accelerator",
                    CACHE_DIR,
                    "{\"graphImportAbi\":\"ggml-fixed-plan-rolling-context-q4-linears-v9\","
                            + "\"requantizeType\":\"Q4_K\"}");
            log.info("PREP_RSS result_len={}", result == null ? -1 : result.toString().length());
            logMemory("after_prepare");
            int pauseSeconds = Integer.getInteger("tensor.g3.preparer.pause.seconds", 0);
            if (pauseSeconds > 0) {
                log.info("PREP_MEMORY pausing_seconds={} for live-process diagnostics", pauseSeconds);
                Thread.sleep(pauseSeconds * 1000L);
                logMemory("after_pause");
            }
        } catch (InvocationTargetException e) {
            throw new RuntimeException("prepare failed", e.getCause());
        } catch (ReflectiveOperationException e) {
            throw new RuntimeException(e);
        } finally {
            running.set(false);
        }

        log.info("PREP_RSS peak_mb={} final_mb={}", peak.get(), rssMb());
    }
}
