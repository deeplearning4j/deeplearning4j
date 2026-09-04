package org.eclipse.deeplearning4j.ggml;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Desktop RSS profile of GGMLModelImport.importModel(File, ConversionOptions.forInference())
 * on the real Qwen3.5-0.8B-Q4_K_M.gguf (532,517,120 bytes, sha256 bd258782...).
 *
 * Reproduces the Pixel 8a ":sdx_model_import" memory death: on device the importer
 * process ramps to ~4.1 GB RSS + 2.6 GB swap inside the "Convert and optimize SDZ"
 * checkpoint even though the GGUF import itself is now chunk-streamed. This test
 * samples /proc/self/status VmRSS every second and reports the peak with phase timing.
 */
@Slf4j
public class TensorG3ConversionRssProfileTest {

    private static final String MODEL_PATH =
            System.getProperty("user.home") + "/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf";

    private static long rssKb() {
        try {
            for (String line : Files.readAllLines(Paths.get("/proc/self/status"))) {
                if (line.startsWith("VmRSS:")) {
                    return Long.parseLong(line.replaceAll("\\D+", ""));
                }
            }
        } catch (Exception ignored) {
        }
        return -1;
    }

    @Test
    public void profileConversionRss() throws Exception {
        File model = new File(MODEL_PATH);
        org.junit.jupiter.api.Assertions.assertTrue(model.isFile(), "model missing: " + model);
        log.info("RSS_PROFILE model={} bytes={}", model, model.length());
        log.info("RSS_PROFILE baseline_rss_mb={}", rssKb() / 1024);

        final AtomicBoolean running = new AtomicBoolean(true);
        final AtomicLong peak = new AtomicLong(rssKb());
        Thread sampler = new Thread(() -> {
            while (running.get()) {
                long r = rssKb();
                if (r > peak.get()) {
                    peak.set(r);
                }
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    return;
                }
            }
        }, "rss-sampler");
        sampler.setDaemon(true);
        sampler.start();

        try {
            ConversionOptions options = ConversionOptions.forInference();
            long t0 = System.currentTimeMillis();
            SameDiff result = GGMLModelImport.importModel(model, options);
            long importMs = System.currentTimeMillis() - t0;
            long postImportRssMb = rssKb() / 1024;

            long vars = result.variables().size();

            MemoryMXBean mx = ManagementFactory.getMemoryMXBean();
            MemoryUsage heap = mx.getHeapMemoryUsage();

            log.info("RSS_PROFILE import_ms={} post_import_rss_mb={} peak_rss_mb={} vars={}",
                    importMs, postImportRssMb, peak.get() / 1024, vars);
            log.info("RSS_PROFILE heap_used_mb={} heap_max_mb={}",
                    heap.getUsed() / (1024 * 1024),
                    heap.getMax() < 0 ? -1 : heap.getMax() / (1024 * 1024));
        } finally {
            running.set(false);
        }

        log.info("RSS_PROFILE final_peak_mb={}", peak.get() / 1024);
    }
}
