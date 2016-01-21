package org.deeplearning4j.ui;

import org.junit.Ignore;
import org.junit.Test;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class ManualTests {

    @Test
    public void testLaunch() throws Exception {

        UiServer server = UiServer.getInstance();

        System.out.println("http://localhost:" + server.getPort()+ "/");

        Thread.sleep(10000000000L);
    }
}
