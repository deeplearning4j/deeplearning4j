package org.deeplearning4j.parallelism.inference.observers;

import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class BatchedInferenceObservableTest {
    @Before
    public void setUp() throws Exception {}

    @After
    public void tearDown() throws Exception {}

    @Test
    public void testVerticalBatch1() throws Exception {
        BatchedInferenceObservable observable = new BatchedInferenceObservable();

        for (int i = 0; i < 32; i++) {
            observable.addInput(new INDArray[]{Nd4j.create(100).assign(i)}, null);
        }

        assertEquals(1, observable.getInputBatches().size());

        INDArray array = observable.getInputBatches().get(0).getFirst()[0];
        assertEquals(2, array.rank());

        log.info("Array shape: {}", Arrays.toString(array.shapeInfoDataBuffer().asInt()));

        for (int i = 0; i < 32; i++) {
            assertEquals((float) i, array.tensorAlongDimension(i, 1).meanNumber().floatValue(), 0.001f);
        }
    }


    @Test
    public void testVerticalBatch2() throws Exception {
        BatchedInferenceObservable observable = new BatchedInferenceObservable();

        for (int i = 0; i < 32; i++) {
            observable.addInput(new INDArray[]{Nd4j.create(1,3, 72, 72).assign(i)}, null);
        }

        assertEquals(1, observable.getInputBatches().size());

        INDArray array = observable.getInputBatches().get(0).getFirst()[0];
        assertEquals(4, array.rank());
        assertEquals(32, array.size(0));

        log.info("Array shape: {}", Arrays.toString(array.shapeInfoDataBuffer().asInt()));

        for (int i = 0; i < 32; i++) {
            assertEquals((float) i, array.tensorAlongDimension(i, 1, 2, 3).meanNumber().floatValue(), 0.001f);
        }
    }

    @Test
    public void testHorizontalBatch1() throws Exception {
        BatchedInferenceObservable observable = new BatchedInferenceObservable();

        for (int i = 0; i < 32; i++) {
            observable.addInput(new INDArray[]{Nd4j.create(3, 72, 72).assign(i), Nd4j.create(3, 100).assign(100 + i)}, null);
        }

        assertEquals(1, observable.getInputBatches().size());

        INDArray[] inputs = observable.getInputBatches().get(0).getFirst();

        INDArray features0 = inputs[0];
        INDArray features1 = inputs[1];

        assertArrayEquals(new long[] {32*3, 72, 72}, features0.shape());
        assertArrayEquals(new long[] {32*3, 100}, features1.shape());

        for (int i = 0; i < 32; i++) {
            for( int j=0; j<3; j++ ) {
                assertEquals((float) i, features0.tensorAlongDimension(3*i + j, 1, 2).meanNumber().floatValue(), 0.001f);
                assertEquals((float) 100 + i, features1.tensorAlongDimension(3*i + j, 1).meanNumber().floatValue(), 0.001f);
            }
        }
    }

    @Test
    public void testTearsBatch1() throws Exception {
        BatchedInferenceObservable observable = new BatchedInferenceObservable();
        INDArray output0 = Nd4j.create(32, 10);
        INDArray output1 = Nd4j.create(32, 15);
        for (int i = 0; i < 32; i++) {
            INDArray t0 = output0.tensorAlongDimension(i, 1).assign(i);
            INDArray t1 = output1.tensorAlongDimension(i, 1).assign(i);
            observable.addInput(new INDArray[]{t0, t1}, null);
        }

        Field f = BatchedInferenceObservable.class.getDeclaredField("outputBatchInputArrays");
        f.setAccessible(true);
        List<int[]> l = new ArrayList<>();
        l.add(new int[]{0,31});
        f.set(observable, l);

        observable.setCounter(32);
        observable.setOutputBatches(Collections.singletonList(new INDArray[]{output0, output1}));

        List<INDArray[]> outputs = observable.getOutputs();

        for (int i = 0; i < 32; i++) {
            assertEquals(2, outputs.get(i).length);

            assertEquals((float) i, outputs.get(i)[0].meanNumber().floatValue(), 0.001f);
            assertEquals((float) i, outputs.get(i)[1].meanNumber().floatValue(), 0.001f);
        }
    }
}
