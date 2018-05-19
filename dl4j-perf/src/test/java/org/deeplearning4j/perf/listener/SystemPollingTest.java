package org.deeplearning4j.perf.listener;

import org.apache.commons.io.FileUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

import java.io.File;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class SystemPollingTest {

    @Rule
    public TemporaryFolder tempDir = new TemporaryFolder();

    @Test
    public void testPolling() throws Exception {
        File tmpDir = tempDir.newFolder();

        SystemPolling systemPolling = new SystemPolling.Builder()
                .outputDirectory(tmpDir).pollEveryMillis(1000)
                .build();
        systemPolling.run();

        Thread.sleep(8000);

        systemPolling.stopPolling();

        File[] files = tmpDir.listFiles();
        assertTrue(files != null && files.length > 0);

        String yaml = FileUtils.readFileToString(files[0]);
        HardwareMetric fromYaml = HardwareMetric.fromYaml(yaml);
        System.out.println(fromYaml);
    }

}
