package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.RandomUtils;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;

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



    @Test
    public void testMultiThreaded_1() throws Exception {
        val numReaders = 4;
        final val tail = new IndexedTail(numReaders);

        val sums = new long[numReaders];
        val readers = new ArrayList<Thread>();
        for (int e = 0; e < numReaders; e++) {
            val f = e;
            val t = new Thread(new Runnable() {
                @Override
                public void run() {
                    sums[f] = 0;
                    while (!tail.isDead()) {
                        while (tail.hasAynthing()) {
                            val updates = Nd4j.create(5, 5);
                            tail.drainTo(updates);
                            val mean = updates.meanNumber().longValue();
                            sums[f] += mean;
                        }
                    }
                }
            });

            t.setName("reader thread " + f);
            t.start();
            readers.add(t);
        }


        int sum = 0;
        for (int e = 0; e < 1000; e++) {
            val array = Nd4j.create(5, 5).assign(e+1);
            Nd4j.getExecutioner().commit();

            sum += (e+1);
            tail.put(array);
        }
        tail.notifyDead();


        for (val t:readers)
            t.join();


        for (int e = 0; e < numReaders; e++)
            assertEquals("Failed for reader [" + e + "]",sums[0], sums[e]);


        assertEquals(0, tail.updatesSize());
    }
}