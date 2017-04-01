package org.nd4j.parameterserver.distributed.logic.sequence;

import org.nd4j.parameterserver.distributed.logic.SequenceProvider;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class BasicSequenceProvider implements SequenceProvider {

    private static final BasicSequenceProvider INSTANCE = new BasicSequenceProvider();
    private static final AtomicLong sequence = new AtomicLong(1);

    private BasicSequenceProvider() {

    }

    public static BasicSequenceProvider getInstance() {
        return INSTANCE;
    }

    @Override
    public Long getNextValue() {
        return sequence.incrementAndGet();
    }
}
