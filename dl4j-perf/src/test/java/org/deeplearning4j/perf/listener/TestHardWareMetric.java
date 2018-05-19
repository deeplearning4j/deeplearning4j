package org.deeplearning4j.perf.listener;

import org.junit.Test;
import oshi.json.SystemInfo;

import static junit.framework.TestCase.assertNotNull;
import static org.junit.Assert.assertEquals;

public class TestHardWareMetric {

    @Test
    public void testHardwareMetric() {
        HardwareMetric hardwareMetric = HardwareMetric.fromSystem(new SystemInfo());
        assertNotNull(hardwareMetric);
        System.out.println(hardwareMetric);

        String yaml = hardwareMetric.toYaml();
        HardwareMetric fromYaml = HardwareMetric.fromYaml(yaml);
        assertEquals(hardwareMetric, fromYaml);
    }

}
