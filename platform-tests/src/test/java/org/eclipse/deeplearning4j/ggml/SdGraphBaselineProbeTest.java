package org.eclipse.deeplearning4j.ggml;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;

import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Probe: what does building a SameDiff with 3116 variables + op-level machinery cost
 * before any weights load (the pre-tensor-1 baseline seen as ~1.3 GB in mem traces)?
 */
@Slf4j
public class SdGraphBaselineProbeTest {

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
    public void probeGraphBaseline() {
        log.info("GRAPH_PROBE baseline={}", rssMb());
        SameDiff sd = SameDiff.create();
        log.info("GRAPH_PROBE after_create={}", rssMb());
        for (int i = 0; i < 3116; i++) {
            sd.var("v" + i, LongShapeDescriptor.fromShape(new long[]{64}, DataType.HALF));
        }
        log.info("GRAPH_PROBE after_3116_vars={}", rssMb());
    }
}
