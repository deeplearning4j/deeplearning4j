package org.nd4j.linalg.jcublas.gpumetrics;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;

/**
 * Created by agibsonccc on 10/10/15.
 */
public class GpuMetricsTest {

    @Test
    public void testMetrics() {
        GpuMetrics metrics = GpuMetrics.blockAndThreads(DataBuffer.Type.FLOAT,1000);
        System.out.println(metrics);
    }

}
