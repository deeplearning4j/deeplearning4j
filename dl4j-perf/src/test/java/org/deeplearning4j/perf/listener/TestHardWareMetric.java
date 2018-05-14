package org.deeplearning4j.perf.listener;

import org.junit.Test;
import oshi.json.SystemInfo;

import static junit.framework.TestCase.assertNotNull;

public class TestHardWareMetric {

    @Test
    public void testHardwareMetric() {
        HardwareMetric hardwareMetric = HardwareMetric.fromSystem(new SystemInfo());
        assertNotNull(hardwareMetric);
        System.out.println(hardwareMetric);
    }

}
