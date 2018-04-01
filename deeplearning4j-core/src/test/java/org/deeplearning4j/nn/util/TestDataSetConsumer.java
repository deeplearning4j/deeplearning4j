package org.deeplearning4j.nn.util;

import lombok.NonNull;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Class that consumes DataSets with specified delays, suitable for testing
 *
 * @author raver119@gmail.com
 */
public class TestDataSetConsumer {
    private DataSetIterator iterator;
    private long delay;
    private AtomicLong count = new AtomicLong(0);
    protected static final Logger logger = LoggerFactory.getLogger(TestDataSetConsumer.class);

    public TestDataSetConsumer(long delay) {
        this.delay = delay;
    }

    public TestDataSetConsumer(@NonNull DataSetIterator iterator, long delay) {
        this.iterator = iterator;
        this.delay = delay;
    }


    /**
     * This method cycles through iterator, whie iterator.hasNext() returns true. After each cycle execution time is simulated either using Thread.sleep() or empty cycle
     *
     * @param consumeWithSleep
     * @return
     */
    public long consumeWhileHasNext(boolean consumeWithSleep) {
        count.set(0);
        if (iterator == null)
            throw new RuntimeException("Can't use consumeWhileHasNext() if iterator isn't set");

        while (iterator.hasNext()) {
            DataSet ds = iterator.next();
            this.consumeOnce(ds, consumeWithSleep);
        }

        return count.get();
    }

    /**
     * This method consumes single DataSet, and spends delay time simulating execution of this dataset
     *
     * @param dataSet
     * @param consumeWithSleep
     * @return
     */
    public long consumeOnce(@NonNull DataSet dataSet, boolean consumeWithSleep) {
        long timeMs = System.currentTimeMillis() + delay;
        while (System.currentTimeMillis() < timeMs) {
            if (consumeWithSleep)
                try {
                    Thread.sleep(delay);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
        }

        count.incrementAndGet();

        if (count.get() % 100 == 0)
            logger.info("Passed {} datasets...", count.get());

        return count.get();
    }

    public long getCount() {
        return count.get();
    }
}
