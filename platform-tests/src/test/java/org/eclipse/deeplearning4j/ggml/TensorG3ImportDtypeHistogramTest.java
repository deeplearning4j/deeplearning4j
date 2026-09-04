package org.eclipse.deeplearning4j.ggml;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.ggml.convert.ConversionOptions;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * dtype/size histogram of the imported graph to localize the ~5.6 GB off-heap resident set.
 */
@Slf4j
public class TensorG3ImportDtypeHistogramTest {

    private static final String MODEL_PATH =
            System.getProperty("user.home") + "/.cache/dl4j-llm-models/Qwen3.5-0.8B-Q4_K_M.gguf";

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

    @Test
    public void histogram() throws Exception {
        File model = new File(MODEL_PATH);
        log.info("DTYPE_HISTO baseline_rss_mb={}", rssMb());

        SameDiff sd = GGMLModelImport.importModel(model, ConversionOptions.forInference());

        log.info("DTYPE_HISTO post_import_rss_mb={}", rssMb());

        Map<String, Long> countByDtype = new HashMap<>();
        Map<String, Long> bytesByDtype = new HashMap<>();
        long totalBytes = 0;
        long largest = 0;
        String largestName = "";

        for (SDVariable v : sd.variables()) {
            INDArray arr = v.getArr(false);
            if (arr == null || arr.isView() || arr.data() == null) {
                continue;
            }
            long bytes = arr.data().length() * arr.dataType().width();
            String dt = arr.dataType().toString();
            countByDtype.merge(dt, 1L, Long::sum);
            bytesByDtype.merge(dt, bytes, Long::sum);
            totalBytes += bytes;
            if (bytes > largest) {
                largest = bytes;
                largestName = v.name() + " " + dt + " " + java.util.Arrays.toString(arr.shape());
            }
        }

        log.info("DTYPE_HISTO total_array_bytes_mb={} largest={}", totalBytes / (1024 * 1024), largestName);
        for (Map.Entry<String, Long> e : bytesByDtype.entrySet()) {
            log.info("DTYPE_HISTO dtype={} count={} bytes_mb={}",
                    e.getKey(), countByDtype.get(e.getKey()), e.getValue() / (1024 * 1024));
        }
        log.info("DTYPE_HISTO final_rss_mb={}", rssMb());
    }
}
