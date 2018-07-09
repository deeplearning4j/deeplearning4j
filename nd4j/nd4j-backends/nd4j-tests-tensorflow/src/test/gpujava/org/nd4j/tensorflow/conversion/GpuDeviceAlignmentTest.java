package org.nd4j.tensorflow.conversion;

import org.junit.Test;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import org.tensorflow.framework.ConfigProto;

import static junit.framework.TestCase.assertTrue;

public class GpuDeviceAlignmentTest {

    @Test
    public void testDeviceAlignment() {
        ConfigProto configProto = GraphRunner.getAlignedWithNd4j();
        assertTrue(configProto.getDeviceFilters(0).contains("gpu"));
    }

}
