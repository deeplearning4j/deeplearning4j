package org.nd4j.linalg.primitives;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.*;

/**
 * Tests for Counter
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class CounterTest {

    @Test
    public void testCounterIncrementAll1() {
        Counter<String> counterA = new Counter<>();

        counterA.incrementCount("A", 1);
        counterA.incrementCount("A", 1);
        counterA.incrementCount("A", 1);



        Counter<String> counterB = new Counter<>();
        counterB.incrementCount("B", 2);
        counterB.incrementCount("B", 2);

        assertEquals(3.0, counterA.getCount("A"), 1e-5);
        assertEquals(4.0, counterB.getCount("B"), 1e-5);

        counterA.incrementAll(counterB);

        assertEquals(3.0, counterA.getCount("A"), 1e-5);
        assertEquals(4.0, counterA.getCount("B"), 1e-5);

        counterA.setCount("B", 234);

        assertEquals(234.0, counterA.getCount("B"), 1e-5);
    }



    @Test
    public void testCounterTopN1() {
        Counter<String> counterA = new Counter<>();

        counterA.incrementCount("A", 1);
        counterA.incrementCount("B", 2);
        counterA.incrementCount("C", 3);
        counterA.incrementCount("D", 4);
        counterA.incrementCount("E", 5);

        counterA.keepTopNElements(4);

        assertEquals(4,counterA.size());

        // we expect element A to be gone
        assertEquals(0.0, counterA.getCount("A"), 1e-5);
        assertEquals(2.0, counterA.getCount("B"), 1e-5);
        assertEquals(3.0, counterA.getCount("C"), 1e-5);
        assertEquals(4.0, counterA.getCount("D"), 1e-5);
        assertEquals(5.0, counterA.getCount("E"), 1e-5);
    }

    @Test
    public void testKeysSorted1() throws Exception {
        Counter<String> counterA = new Counter<>();

        counterA.incrementCount("A", 1);
        counterA.incrementCount("B", 2);
        counterA.incrementCount("C", 3);
        counterA.incrementCount("D", 4);
        counterA.incrementCount("E", 5);

        assertEquals("E", counterA.argMax());

        List<String> list = counterA.keySetSorted();

        assertEquals(5, list.size());

        assertEquals("E", list.get(0));
        assertEquals("D", list.get(1));
        assertEquals("C", list.get(2));
        assertEquals("B", list.get(3));
        assertEquals("A", list.get(4));
    }
    
    @Test
    public void testCounterTotal() {
        Counter<String> counter = new Counter<>();

        counter.incrementCount("A", 1);
        counter.incrementCount("B", 1);
        counter.incrementCount("C", 1);

        assertEquals(3.0, counter.totalCount(), 1e-5);
        
        counter.setCount("B", 234);

        assertEquals(236.0, counter.totalCount(), 1e-5);
        
        counter.setCount("D", 1);
        
        assertEquals(237.0, counter.totalCount(), 1e-5);
        
        counter.removeKey("B");
        
        assertEquals(3.0, counter.totalCount(), 1e-5);
        
    }
    
}