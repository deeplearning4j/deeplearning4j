package org.nd4j.linalg;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.util.Pair;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class AveragingTests extends BaseNd4jTest {
    private final int THREADS = 16;
    private final int LENGTH = 5120000 * 4;

    DataBuffer.Type initialType;

    public AveragingTests(Nd4jBackend backend) {
        super(backend);
        this.initialType = Nd4j.dataType();
    }

    @Before
    public void setUp() {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
    }

    @After
    public void shutUp() {
        DataTypeUtil.setDTypeForContext(initialType);
    }



    @Test
    public void testSingleDeviceAveraging() throws Exception {
        INDArray array1 = Nd4j.valueArrayOf(LENGTH, 1.0);
        INDArray array2 = Nd4j.valueArrayOf(LENGTH, 2.0);
        INDArray array3 = Nd4j.valueArrayOf(LENGTH, 3.0);
        INDArray array4 = Nd4j.valueArrayOf(LENGTH, 4.0);
        INDArray array5 = Nd4j.valueArrayOf(LENGTH, 5.0);
        INDArray array6 = Nd4j.valueArrayOf(LENGTH, 6.0);
        INDArray array7 = Nd4j.valueArrayOf(LENGTH, 7.0);
        INDArray array8 = Nd4j.valueArrayOf(LENGTH, 8.0);
        INDArray array9 = Nd4j.valueArrayOf(LENGTH, 9.0);
        INDArray array10 = Nd4j.valueArrayOf(LENGTH, 10.0);
        INDArray array11 = Nd4j.valueArrayOf(LENGTH, 11.0);
        INDArray array12 = Nd4j.valueArrayOf(LENGTH, 12.0);
        INDArray array13 = Nd4j.valueArrayOf(LENGTH, 13.0);
        INDArray array14 = Nd4j.valueArrayOf(LENGTH, 14.0);
        INDArray array15 = Nd4j.valueArrayOf(LENGTH, 15.0);
        INDArray array16 = Nd4j.valueArrayOf(LENGTH, 16.0);


        long time1 = System.currentTimeMillis();
        INDArray arrayMean = Nd4j.averageAndPropagate(new INDArray[] {array1, array2, array3, array4, array5, array6, array7, array8, array9, array10, array11, array12, array13, array14, array15, array16});
        long time2 = System.currentTimeMillis();
        System.out.println("Execution time: " + (time2 - time1));

        assertNotEquals(null, arrayMean);

        assertEquals(8.5f, arrayMean.getFloat(12), 0.1f);
        assertEquals(8.5f, arrayMean.getFloat(150), 0.1f);
        assertEquals(8.5f, arrayMean.getFloat(475), 0.1f);


        assertEquals(8.5f, array1.getFloat(475), 0.1f);
        assertEquals(8.5f, array2.getFloat(475), 0.1f);
        assertEquals(8.5f, array3.getFloat(475), 0.1f);
        assertEquals(8.5f, array5.getFloat(475), 0.1f);
        assertEquals(8.5f, array16.getFloat(475), 0.1f);


        assertEquals(8.5, arrayMean.meanNumber().doubleValue(), 0.01);
        assertEquals(8.5, array1.meanNumber().doubleValue(), 0.01);
        assertEquals(8.5, array2.meanNumber().doubleValue(), 0.01);

        assertEquals(arrayMean, array16);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
