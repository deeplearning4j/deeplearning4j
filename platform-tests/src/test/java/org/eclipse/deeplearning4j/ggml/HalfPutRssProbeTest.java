package org.eclipse.deeplearning4j.ggml;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Minimal probe: interval-put chunk writes into a HALF array — RSS attribution only.
 * 254,279,680 elements (Qwen3.5 token_embd shape [1024, 248320]).
 */
@Slf4j
public class HalfPutRssProbeTest {

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

    private static void logRss(String stage) {
        log.info("PUT_PROBE stage={} rssMb={}", stage, rssMb());
    }

    @Test
    public void probeIntervalPutHalf() {
        long rows = 1024;
        long cols = 248320;
        logRss("baseline");

        INDArray output = Nd4j.createUninitialized(DataType.HALF, rows, cols);
        logRss("after_output_alloc_half_970MB");

        INDArray chunkArr = Nd4j.createUninitialized(DataType.HALF, 1, 8_388_608);
        logRss("after_chunk_alloc_32MB");

        // Fill chunk with data (mimics dequantized chunk)
        for (int i = 0; i < 4; i++) {
            chunkArr.putScalar(i, (float) i);
        }

        long offset = 0;
        long total = rows * cols;
        while (offset < total) {
            long n = Math.min(8_388_608, total - offset);
            long row = offset / cols;
            long col = offset % cols;
            long nThisRow = Math.min(n, cols - col);
            output.put(new INDArrayIndex[]{
                    NDArrayIndex.point(row),
                    NDArrayIndex.interval(col, col + nThisRow)},
                    chunkArr.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nThisRow)));
            offset += nThisRow;
        }
        logRss("after_full_put_loop");

        output.close();
        chunkArr.close();
        logRss("after_close");
    }
}
