package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;

import static org.junit.Assert.*;

@Slf4j
public class IndexedTailTest {

    @Test
    public void testSingleThreaded_1() throws Exception {
        val tail = new IndexedTail(1);

        for (int e = 0; e < 100; e++) {

        }
    }
}