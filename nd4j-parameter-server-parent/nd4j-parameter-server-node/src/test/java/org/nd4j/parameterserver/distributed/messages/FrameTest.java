package org.nd4j.parameterserver.distributed.messages;

import org.agrona.concurrent.UnsafeBuffer;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.enums.NodeRole;
import org.nd4j.parameterserver.distributed.logic.Clipboard;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.transport.Transport;

import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class FrameTest {
    @Before
    public void setUp() throws Exception {

    }

    /**
     * Simple test for Frame functionality
     */
    @Test
    public void testFrame1() {
        final AtomicInteger count = new AtomicInteger(0);

        Frame<VoidMessage> frame = new Frame<>();

        for(int i = 0; i < 10; i++) {
            frame.stackMessage(new VoidMessage() {
                @Override
                public long getTaskId() {
                    return 0;
                }

                @Override
                public int getMessageType() {
                    return 0;
                }

                @Override
                public byte[] asBytes() {
                    return new byte[0];
                }

                @Override
                public UnsafeBuffer asUnsafeBuffer() {
                    return null;
                }

                @Override
                public void attachContext(Configuration configuration, TrainingDriver<? extends TrainingMessage> trainer, Clipboard clipboard, Transport transport, Storage storage, NodeRole role, short shardIndex) {
                    // no-op intentionally
                }

                @Override
                public void processMessage() {
                    count.incrementAndGet();
                }
            });
        }

        assertEquals(10, frame.size());

        frame.processMessage();

        assertEquals(10, count.get());
    }


}