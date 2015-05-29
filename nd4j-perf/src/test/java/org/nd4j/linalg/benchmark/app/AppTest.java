package org.nd4j.linalg.benchmark.app;

import org.junit.Test;

/**
 * @author Adam Gibson
 */
public class AppTest {
    @Test
    public void testApp() throws Exception {
        BenchmarkRunnerApp app = new BenchmarkRunnerApp();
        app.doMain(new String[]{"-n","100"});
    }

}
