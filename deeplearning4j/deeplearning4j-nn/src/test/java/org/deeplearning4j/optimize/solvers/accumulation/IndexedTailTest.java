package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
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

        assertEquals(0, tail.updatesSize());
    }

    @Test
    public void testSingleThreaded_2() throws Exception {
        val tail = new IndexedTail(1);

        for (int e = 0; e < 100; e++) {
            int numUpdates = RandomUtils.nextInt(1, 10);
            int sum = 0;

            for (int f = 1; f <= numUpdates; f++) {
                sum += f;
                val orig = Nd4j.create(5, 5).assign(f);
                tail.put(orig);
            }
            Nd4j.getExecutioner().commit();

            assertTrue(tail.hasAynthing());

            val temp = Nd4j.create(5, 5);
            val status = tail.drainTo(temp);

            assertTrue(status);
            assertEquals(sum, temp.meanNumber().intValue());
        }

        assertEquals(0, tail.updatesSize());
    }


    @Test
    public void testPseudoMultiThreaded_1() throws Exception {
        val tail = new IndexedTail(2);

        for (int e = 0; e < 100; e++) {
            // putting in one thread
            val orig = Nd4j.create(5, 5).assign(e);
            tail.put(orig);
            Nd4j.getExecutioner().commit();

            for (int t = 0; t < 2; t++) {
                assertTrue(tail.hasAynthing(t));

                val temp = Nd4j.create(5, 5);
                val status = tail.drainTo(t, temp);

                assertTrue(status);
                assertArrayEquals(orig.shape(), temp.shape());
                assertEquals(orig, temp);
            }
        }

        assertEquals(0, tail.updatesSize());
    }
}