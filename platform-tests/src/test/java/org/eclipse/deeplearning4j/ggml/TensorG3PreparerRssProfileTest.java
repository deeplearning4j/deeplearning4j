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

    private static final class PathFingerprint {
        static void log(java.net.URL location) throws Exception {
            java.nio.file.Path path = Paths.get(location.toURI());
            if (!Files.isRegularFile(path)) return;
            java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
            try (java.io.InputStream input = Files.newInputStream(path)) {
                byte[] buffer = new byte[65536];
                int count;
                while ((count = input.read(buffer)) >= 0) digest.update(buffer, 0, count);
            }
            StringBuilder hex = new StringBuilder();
            for (byte value : digest.digest()) hex.append(String.format("%02x", value & 255));
            log.info("PREP_ARTIFACT sha256={} path={}", hex, path);
        }
    }

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
        int trimAtSeconds = Integer.getInteger("tensor.g3.preparer.trimOnceSeconds", 0);
        final NativeOps trimOps;
        if (trimAtSeconds > 0) {
            org.junit.jupiter.api.Assertions.assertTrue(System.getProperty("os.name").equals("Linux"));
            org.junit.jupiter.api.Assertions.assertTrue(Nd4j.getBackend().getClass().getName().contains(".cpu."),
                    "Allocator-only diagnostic must never trim a live device pool");
            trimOps = Nd4j.getNativeOps();
        } else {
            trimOps = null;
        }
        AtomicBoolean trimmed = new AtomicBoolean(false);
        java.util.concurrent.atomic.AtomicReference<Throwable> samplerFailure = new java.util.concurrent.atomic.AtomicReference<>();
        log.info("PREP_RSS baseline_mb={}", rssMb());

        final AtomicBoolean running = new AtomicBoolean(true);
        final AtomicLong peak = new AtomicLong(rssMb());
        long started = System.nanoTime();
        Thread sampler = new Thread(() -> {
            int samples = 0;
            while (running.get()) {
                long r = rssMb();
                if (r > peak.get()) {
                    peak.set(r);
                }
                if (samples++ % 10 == 0) {
                    MemoryUsage heap = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage();
                    log.info("PREP_RSS elapsed_s={} rss_mb={} peak_mb={} heap_used_mb={} "
                                    + "heap_committed_mb={} javacpp_total_mb={}",
                            (System.nanoTime() - started) / 1_000_000_000L, r, peak.get(),
                            megabytes(heap.getUsed()), megabytes(heap.getCommitted()),
                            megabytes(Pointer.totalBytes()));
                }
                if (trimOps != null && (System.nanoTime() - started) / 1_000_000_000L >= trimAtSeconds
                        && trimmed.compareAndSet(false, true)) {
                    try {
                        long beforeTrim = rssMb();
                        long trimStart = System.nanoTime();
                        // CPU Linux implementation is thread-safe malloc_trim(0):
                        // return unused allocator pages, never close arrays or reset plans.
                        trimOps.trimMemoryPool(0);
                        log.info("PREP_TRIM before_mb={} after_mb={} elapsed_ms={}", beforeTrim, rssMb(),
                                (System.nanoTime() - trimStart) / 1_000_000L);
                    } catch (Throwable failure) {
                        samplerFailure.set(failure);
                        return;
                    }
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
            for (Class<?> type : new Class<?>[]{cls,
                    org.nd4j.dsp.model.SdxTensorG3Q4Calibration.class,
                    org.eclipse.deeplearning4j.llm.generation.GenerationPipeline.class,
                    org.nd4j.ggml.convert.GGMLToSameDiffConverter.class}) {
                java.net.URL location = type.getProtectionDomain().getCodeSource().getLocation();
                log.info("PREP_ARTIFACT class={} location={}", type.getName(), location);
                PathFingerprint.log(location);
            }
            Method prepare = cls.getDeclaredMethod("prepare",
                    String.class, String.class, String.class, String.class, String.class);
            prepare.setAccessible(true);
            Object result = prepare.invoke(null,
                    MODEL_PATH,
                    TOKENIZER_DIR,
                    "android-arm64-nnapi-accelerator",
                    CACHE_DIR,
                    System.getProperty("tensor.g3.preparer.options",
                            "{\"graphImportAbi\":\"ggml-fixed-plan-rolling-context-q4-linears-v9\","
                                    + "\"requantizeType\":\"Q4_K\"}"));
            log.info("PREP_RSS result_len={}", result == null ? -1 : result.toString().length());
            logMemory("after_prepare");
            if (trimOps != null) {
                org.junit.jupiter.api.Assertions.assertTrue(trimmed.get(), "Run ended before allocator probe");
                org.junit.jupiter.api.Assertions.assertNull(samplerFailure.get(), "Allocator probe failed");
            }
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
            sampler.interrupt();
            sampler.join(2000);
            log.info("PREP_RSS peak_mb={} final_mb={}", peak.get(), rssMb());
        }
    }
}
