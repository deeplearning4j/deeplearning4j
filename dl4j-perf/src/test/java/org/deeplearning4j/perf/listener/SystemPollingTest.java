package org.deeplearning4j.perf.listener;

import org.apache.commons.io.FileUtils;
import org.junit.Test;

import java.io.File;

import static org.junit.Assert.assertEquals;

public class SystemPollingTest {

    @Test
    public void testPolling() throws Exception {
        File tmpDir = new File("tmpdir-polling");
        if(tmpDir.exists()) {
            FileUtils.deleteDirectory(tmpDir);
        }

        if(!tmpDir.exists()) {
            tmpDir.mkdirs();
        }

        SystemPolling systemPolling = new SystemPolling.Builder()
                .outputDirectory(tmpDir).pollEveryMillis(1000)
                .build();
        systemPolling.run();

        Thread.sleep(6000);

        systemPolling.stopPolling();

        assertEquals(4,tmpDir.list().length);
        FileUtils.deleteDirectory(tmpDir);
    }

}
