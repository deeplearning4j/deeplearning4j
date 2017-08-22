package org.deeplearning4j.spark.parameterserver.iterators;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class VirtualIteratorTest {
    @Before
    public void setUp() throws Exception {
        //
    }

    @Test
    public void testIteration1() throws Exception {
        List<Integer> integers = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            integers.add(i);
        }

        VirtualIterator<Integer> virt = new VirtualIterator<>(integers.iterator());

        int cnt = 0;
        while (virt.hasNext()) {
            Integer n = virt.next();
            assertEquals(cnt, n.intValue());
            cnt++;
        }


        assertEquals(100, cnt);
    }
}
