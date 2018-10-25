package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

@Slf4j
public class IndexedTailTest {

    @Test
    public void testSingleThreaded_1() throws Exception {
        val tail = new IndexedTail(1);

        for (int e = 0; e < 100; e++) {
            val orig = Nd4j.create(5, 5).assign(e);
            tail.put(orig);
            Nd4j.getExecutioner().commit();

            assertTrue(tail.hasAynthing());

            val temp = Nd4j.create(5, 5);
            val status = tail.drainTo(temp);

            assertTrue(status);
            assertArrayEquals(orig.shape(), temp.shape());
            assertEquals(orig, temp);
        }
    }
}