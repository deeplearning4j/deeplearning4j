package org.nd4j.instrumentation.server;

import org.junit.Test;

/**
 * Created by agibsonccc on 3/21/15.
 */
public class RunInstrumentationApplication {

    @Test
    public void testInstrumentationApplication() throws Exception {
        InstrumentationApplication app = new InstrumentationApplication();
        app.start();
        app.stop();
    }


}
