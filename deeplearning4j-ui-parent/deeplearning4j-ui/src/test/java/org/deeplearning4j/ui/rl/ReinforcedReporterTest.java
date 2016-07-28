package org.deeplearning4j.ui.rl;

import org.deeplearning4j.ui.UiServer;
import org.junit.Ignore;
import org.junit.Test;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class ReinforcedReporterTest {

    @Test
    public void testReporter1() throws Exception {

        UiServer.createServer();

        System.out.println("Server: http://localhost:" + UiServer.getInstance().getPort() + "/rl");

        Thread.sleep(1000);

        ReinforcedReporter reporter = new ReinforcedReporter(UiServer.getInstance().getConnectionInfo());

        for (int x = 0; x < 10; x++ ) {
            reporter.report(x, x);
        }

        Thread.sleep(100000000);
    }
}