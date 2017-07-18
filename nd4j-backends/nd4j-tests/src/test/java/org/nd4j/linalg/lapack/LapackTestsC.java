package org.nd4j.linalg.lapack;

import lombok.extern.slf4j.Slf4j;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@RunWith(Parameterized.class)
public class LapackTestsC extends BaseNd4jTest {
    DataBuffer.Type initialType;

    public LapackTestsC(Nd4jBackend backend) {
        super(backend);
        initialType = Nd4j.dataType();
    }

    @Before
    public void setUp() {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
    }

    @After
    public void after() {
        Nd4j.setDataType(initialType);
    }

    @Test
    public void testGetRF1DifferentOrders() throws Exception {
        INDArray a = Nd4j.linspace(1, 9, 9).reshape(3, 3);
        INDArray exp = Nd4j.create(new double[] {7.0, 8.0, 9.0, 0.14285715, 0.85714287, 1.7142857, 0.5714286, 0.5, 0.0},
                        new int[] {3, 3}, 'c');

        INDArray r = Nd4j.getNDArrayFactory().lapack().getrf(a);

        assertEquals(exp, a);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
