package org.nd4j.linalg.shape.concat.padding;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * @author Adam Gibson
 */
@RunWith(Parameterized.class)
public class PaddingTests extends BaseNd4jTest {

    public PaddingTests(Nd4jBackend backend) {
        super(backend);
    }



    @Test
    public void testAppend() {
        INDArray appendTo = Nd4j.ones(3, 3);
        INDArray ret = Nd4j.append(appendTo, 3, 1, -1);
        assertArrayEquals(new long[] {3, 6}, ret.shape());

        INDArray linspace = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray otherAppend = Nd4j.append(linspace, 3, 1.0, -1);
        INDArray assertion = Nd4j.create(new double[][] {{1, 3, 1, 1, 1}, {2, 4, 1, 1, 1}});

        assertEquals(assertion, otherAppend);


    }

    @Test
    public void testPrepend() {
        INDArray appendTo = Nd4j.ones(3, 3);
        INDArray ret = Nd4j.append(appendTo, 3, 1, -1);
        assertArrayEquals(new long[] {3, 6}, ret.shape());

        INDArray linspace = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray assertion = Nd4j.create(new double[][] {{1, 1, 1, 1, 3}, {1, 1, 1, 2, 4}});

        INDArray prepend = Nd4j.prepend(linspace, 3, 1.0, -1);
        assertEquals(assertion, prepend);

    }



    @Test
    public void testPad() {
        INDArray start = Nd4j.linspace(1, 9, 9).reshape(3, 3);
        INDArray ret = Nd4j.pad(start, new int[] {5, 5}, Nd4j.PadMode.CONSTANT);
        double[][] data = new double[][] {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.},
                        {0, 0, 0, 0, 0, 1, 4, 7, 0, 0, 0, 0, 0.}, {0, 0, 0, 0, 0, 2, 5, 8, 0, 0, 0, 0, 0.},
                        {0, 0, 0, 0, 0, 3, 6, 9, 0, 0, 0, 0, 0.}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.},
                        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.}};
        INDArray assertion = Nd4j.create(data);
        assertEquals(assertion, ret);


    }


    @Override
    public char ordering() {
        return 'f';
    }
}
