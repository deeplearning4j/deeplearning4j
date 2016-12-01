package org.nd4j.linalg.profiling;

import org.junit.Test;
import org.nd4j.linalg.profiler.data.StackAggregator;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class StackAggregatorTests {


    @Test
    public void testBasicBranching1() {
        StackAggregator aggregator = new StackAggregator();

        aggregator.incrementCount();

        aggregator.incrementCount();

        assertEquals(2, aggregator.getTotalEventsNumber());
        assertEquals(2, aggregator.getUniqueBranchesNumber());
    }

    @Test
    public void testBasicBranching2() {
        StackAggregator aggregator = new StackAggregator();

        for (int i = 0; i < 10; i++) {
            aggregator.incrementCount();
        }

        assertEquals(10, aggregator.getTotalEventsNumber());
        assertEquals(1, aggregator.getUniqueBranchesNumber());
    }
}
