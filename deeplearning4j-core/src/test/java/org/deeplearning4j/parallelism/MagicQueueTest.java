package org.deeplearning4j.parallelism;

import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class MagicQueueTest {
    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void addDataSet1() throws Exception {
        MagicQueue queue = new MagicQueue.Builder().build();

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        queue.add(new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f})));
        queue.add(new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f})));
        queue.add(new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f})));
        queue.add(new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f})));
        queue.add(new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f})));
        queue.add(new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f})));
        queue.add(new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f})));
        queue.add(new DataSet(Nd4j.create(new float[]{1f,2f,3f}), Nd4j.create(new float[]{1f,2f,3f})));

        Thread.sleep(1100);

        assertEquals(8 / numDevices, queue.size());
    }



}