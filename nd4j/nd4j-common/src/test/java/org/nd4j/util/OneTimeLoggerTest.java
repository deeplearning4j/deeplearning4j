package org.nd4j.util;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class OneTimeLoggerTest {

    @Test
    public void testLogger1() throws Exception {
        OneTimeLogger.info(log, "Format: {}; Pew: {};", 1, 2);
    }

    @Test
    public void testBuffer1() throws Exception {
        assertTrue(OneTimeLogger.isEligible("Message here"));

        assertFalse(OneTimeLogger.isEligible("Message here"));

        assertTrue(OneTimeLogger.isEligible("Message here 23"));
    }
}
