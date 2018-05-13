package org.deeplearning4j.perf.listener;

import org.junit.Test;

import java.io.File;

public class TestSystemInfoPrintListener {

    @Test
    public void testListener() {
        SystemInfoPrintListener systemInfoPrintListener = SystemInfoPrintListener.builder()
                .printOnEpochStart(true).printOnEpochEnd(true)
                .build();

        File tmpFile = new File("tmpfile-log.txt");
        SystemInfoFilePrintListener systemInfoFilePrintListener = SystemInfoFilePrintListener.builder()
                .printOnEpochStart(true).printOnEpochEnd(true).printFileTarget(tmpFile)
                .build();
        tmpFile.deleteOnExit();
    }

}
