package org.deeplearning4j.linalg.jcublas;


import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * NDArrayTests
 * @author Adam Gibson
 */
public class JCublasNDArrayTests extends org.deeplearning4j.linalg.api.test.NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(JCublasNDArrayTests.class);

    @Test
    public void testMmul2() {

        NDArrays.factory().setOrder('c');

        float[] data = NDArrays.linspace(1, 10, 10).data();
        INDArray n = NDArrays.create(data, new int[]{10});
        INDArray m = NDArrays.create(data, new int[]{10});
        INDArray e = n.mmul(n.transpose());
    }

}