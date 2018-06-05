package org.nd4j.parameterserver.distributed.logic;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.Timeout;
import org.nd4j.parameterserver.distributed.logic.completion.FrameCompletionHandler;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class FrameCompletionHandlerTest {
    @Before
    public void setUp() throws Exception {

    }

    @Rule
    public Timeout globalTimeout = Timeout.seconds(30);

    /**
     * This test emulates 2 frames being processed at the same time
     * @throws Exception
     */
    @Test
    public void testCompletion1() throws Exception {
        FrameCompletionHandler handler = new FrameCompletionHandler();
        long[] frames = new long[] {15L, 17L};
        long[] originators = new long[] {123L, 183L};
        for (Long originator : originators) {
            for (Long frame : frames) {
                for (int e = 1; e <= 512; e++) {
                    handler.addHook(originator, frame, (long) e);
                }
            }

            for (Long frame : frames) {
                for (int e = 1; e <= 512; e++) {
                    handler.notifyFrame(originator, frame, (long) e);
                }
            }
        }


        for (Long originator : originators) {
            for (Long frame : frames) {
                assertEquals(true, handler.isCompleted(originator, frame));
            }
        }
    }

}
