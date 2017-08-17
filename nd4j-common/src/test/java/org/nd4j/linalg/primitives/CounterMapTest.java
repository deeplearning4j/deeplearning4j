package org.nd4j.linalg.primitives;

import org.junit.Test;

import java.util.Iterator;

import static org.junit.Assert.*;

/**
 * CounterMap tests
 *
 * @author raver119@gmail.com
 */
public class CounterMapTest {

    @Test
    public void testIterator() {
        CounterMap<Integer, Integer> counterMap = new CounterMap<>();

        counterMap.incrementCount(0, 0, 1);
        counterMap.incrementCount(0, 1, 1);
        counterMap.incrementCount(0, 2, 1);
        counterMap.incrementCount(1, 0, 1);
        counterMap.incrementCount(1, 1, 1);
        counterMap.incrementCount(1, 2, 1);

        Iterator<Pair<Integer, Integer>> iterator = counterMap.getIterator();

        Pair<Integer, Integer> pair = iterator.next();

        assertEquals(0, pair.getFirst().intValue());
        assertEquals(0, pair.getSecond().intValue());

        pair = iterator.next();

        assertEquals(0, pair.getFirst().intValue());
        assertEquals(1, pair.getSecond().intValue());

        pair = iterator.next();

        assertEquals(0, pair.getFirst().intValue());
        assertEquals(2, pair.getSecond().intValue());

        pair = iterator.next();

        assertEquals(1, pair.getFirst().intValue());
        assertEquals(0, pair.getSecond().intValue());

        pair = iterator.next();

        assertEquals(1, pair.getFirst().intValue());
        assertEquals(1, pair.getSecond().intValue());

        pair = iterator.next();

        assertEquals(1, pair.getFirst().intValue());
        assertEquals(2, pair.getSecond().intValue());


        assertFalse(iterator.hasNext());
    }


    @Test
    public void testIncrementAll() {
        CounterMap<Integer, Integer> counterMapA = new CounterMap<>();

        counterMapA.incrementCount(0, 0, 1);
        counterMapA.incrementCount(0, 1, 1);
        counterMapA.incrementCount(0, 2, 1);
        counterMapA.incrementCount(1, 0, 1);
        counterMapA.incrementCount(1, 1, 1);
        counterMapA.incrementCount(1, 2, 1);

        CounterMap<Integer, Integer> counterMapB = new CounterMap<>();

        counterMapB.incrementCount(1, 1, 1);
        counterMapB.incrementCount(2, 1, 1);

        counterMapA.incrementAll(counterMapB);

        assertEquals(2.0, counterMapA.getCount(1,1), 1e-5);
        assertEquals(1.0, counterMapA.getCount(2,1), 1e-5);
        assertEquals(1.0, counterMapA.getCount(0,0), 1e-5);


        assertEquals(7, counterMapA.totalSize());


        counterMapA.setCount(2, 1, 17);

        assertEquals(17.0, counterMapA.getCount(2, 1), 1e-5);
    }
}