package org.deeplearning4j.parallelism.inference.observers;

import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

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
            observable.setInput(Nd4j.create(100).assign(i));
        }

        assertEquals(1, observable.getInput().length);

        INDArray array = observable.getInput()[0];
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
            observable.setInput(Nd4j.create(3, 72, 72).assign(i));
        }

        assertEquals(1, observable.getInput().length);

        INDArray array = observable.getInput()[0];
        assertEquals(4, array.rank());
        assertEquals(32, array.shape()[0]);

        log.info("Array shape: {}", Arrays.toString(array.shapeInfoDataBuffer().asInt()));

        for (int i = 0; i < 32; i++) {
            assertEquals((float) i, array.tensorAlongDimension(i, 1, 2, 3).meanNumber().floatValue(), 0.001f);
        }
    }

    @Test
    public void testHorizontalBatch1() throws Exception {
        BatchedInferenceObservable observable = new BatchedInferenceObservable();

        for (int i = 0; i < 32; i++) {
            observable.setInput(Nd4j.create(3, 72, 72).assign(i), Nd4j.create(100, 100).assign(100 + i));
        }

        assertEquals(2, observable.getInput().length);

        INDArray[] inputs = observable.getInput();

        INDArray features0 = inputs[0];
        INDArray features1 = inputs[1];

        assertArrayEquals(new int[] {32, 3, 72, 72}, features0.shape());
        assertArrayEquals(new int[] {32, 100, 100}, features1.shape());

        for (int i = 0; i < 32; i++) {
            assertEquals((float) i, features0.tensorAlongDimension(i, 1, 2, 3).meanNumber().floatValue(), 0.001f);
            assertEquals((float) 100 + i, features1.tensorAlongDimension(i, 1, 2).meanNumber().floatValue(), 0.001f);
        }
    }

    @Test
    public void testTearsBatch1() throws Exception {
        BatchedInferenceObservable observable = new BatchedInferenceObservable();
        INDArray output0 = Nd4j.create(32, 10);
        INDArray output1 = Nd4j.create(32, 15);
        for (int i = 0; i < 32; i++) {
            output0.tensorAlongDimension(i, 1).assign(i);
            output1.tensorAlongDimension(i, 1).assign(i);
        }

        observable.setCounter(32);
        observable.setOutput(output0, output1);

        List<INDArray[]> outputs = observable.getOutputs();

        for (int i = 0; i < 32; i++) {
            assertEquals(2, outputs.get(i).length);

            assertEquals((float) i, outputs.get(i)[0].meanNumber().floatValue(), 0.001f);
            assertEquals((float) i, outputs.get(i)[1].meanNumber().floatValue(), 0.001f);
        }
    }
}
