package org.deeplearning4j.parallelism;

import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class AsyncIteratorTest {

    @Test
    public void hasNext() throws Exception {
        ArrayList<Integer> integers = new ArrayList<>();
        for (int x = 0; x < 100000; x++ ) {
            integers.add(x);
        }

        AsyncIterator<Integer> iterator = new AsyncIterator<Integer>(integers.iterator(), 512);
        int cnt = 0;
        while (iterator.hasNext()) {
            iterator.next();
            cnt++;
        }

        assertEquals(integers.size(), cnt);
    }

}