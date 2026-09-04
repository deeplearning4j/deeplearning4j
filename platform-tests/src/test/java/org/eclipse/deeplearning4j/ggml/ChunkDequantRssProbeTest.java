package org.eclipse.deeplearning4j.ggml;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.ggml.format.GGMLDataType;
import org.nd4j.ggml.quantization.DequantizerFactory;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * Probe: repeated dequantizeToArray chunks (the streaming loop's inner op) — RSS attribution.
 * 31 chunks x 8M elements = 254M elements, matching Qwen3.5 token_embd (Q4_K).
 */
@Slf4j
public class ChunkDequantRssProbeTest {

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
    public void probeChunkDequantRss() {
        int blockSize = 256;      // Q4_K: 256 elements/block
        int bytesPerBlock = 144;  // Q4_K: 144 bytes/block
        long chunkElements = 8_388_608L;
        int chunks = 31;

        log.info("DEQ_PROBE baseline_mb={}", rssMb());

        byte[] chunk = new byte[(int) (chunkElements / blockSize * bytesPerBlock)];
        for (int i = 0; i < 100; i++) {
            chunk[i] = (byte) (i * 7 + 3);
        }

        for (int c = 0; c < chunks; c++) {
            INDArray out = DequantizerFactory.dequantizeToArray(
                    Arrays.copyOf(chunk, chunk.length),
                    GGMLDataType.GGML_TYPE_Q4_K,
                    new long[]{chunkElements}, DataType.HALF);
            out.close();
            if (c == 0 || c == chunks - 1) {
                log.info("DEQ_PROBE after_chunk={} mb={}", c, rssMb());
            }
        }

        log.info("DEQ_PROBE after_all_mb={}", rssMb());
    }
}
