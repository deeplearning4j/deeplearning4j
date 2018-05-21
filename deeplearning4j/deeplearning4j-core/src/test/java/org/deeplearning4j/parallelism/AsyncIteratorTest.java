package org.deeplearning4j.parallelism;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;

import java.util.ArrayList;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class AsyncIteratorTest extends BaseDL4JTest {

    @Test
    public void hasNext() throws Exception {
        ArrayList<Integer> integers = new ArrayList<>();
        for (int x = 0; x < 100000; x++) {
            integers.add(x);
        }

        AsyncIterator<Integer> iterator = new AsyncIterator<>(integers.iterator(), 512);
        int cnt = 0;
        Integer val = null;
        while (iterator.hasNext()) {
            val = iterator.next();
            assertEquals(cnt, val.intValue());
            cnt++;
        }

        System.out.println("Last val: " + val);

        assertEquals(integers.size(), cnt);
    }

}
