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
    public void testSum() {
        /*
        INDArray n = NDArrays.create(NDArrays.linspace(1, 8, 8).data(), new int[]{2, 2, 2});
        INDArray test = NDArrays.create(new double[]{3,7,11,15},new int[]{2,2});
        INDArray sum = n.sum(n.shape().length - 1);
        assertEquals(test,sum);
        */
        INDArray n = NDArrays.create(NDArrays.linspace(1, 8, 8).data(), new int[]{2, 2, 2});
        JCublasNDArray d = (JCublasNDArray) n;
        n = d.norm2(0);

    }

}