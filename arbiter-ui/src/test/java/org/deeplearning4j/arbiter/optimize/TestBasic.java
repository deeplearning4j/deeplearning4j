package org.deeplearning4j.arbiter.optimize;

import org.deeplearning4j.ui.api.UIServer;
import org.junit.Test;

/**
 * Created by Alex on 19/07/2017.
 */
public class TestBasic {

    @Test
    public void testBasic() throws Exception {

        UIServer.getInstance();

        Thread.sleep(100000);
    }

}
