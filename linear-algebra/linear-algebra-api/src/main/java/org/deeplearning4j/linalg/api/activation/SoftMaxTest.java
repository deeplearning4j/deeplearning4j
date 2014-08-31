package org.deeplearning4j.linalg.api.activation;

import static org.junit.Assert.*;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Test for softmax function
 *
 * @author Adam Gibson
 */
public abstract class SoftMaxTest {

    private static Logger log = LoggerFactory.getLogger(SoftMaxTest.class);

    @Test
    public void testSoftMax() {
        NDArrays.factory().setOrder('f');
        INDArray test = NDArrays.linspace(1,4,4).reshape(2,2);
        INDArray softMaxColumns = Activations.softmax().apply(test);
        INDArray softMaxRows = Activations.softMaxRows().apply(test);
        assertEquals(2,softMaxColumns.sum(Integer.MAX_VALUE).get(0),1e-1);
        assertEquals(2,softMaxRows.sum(Integer.MAX_VALUE).get(0),1e-1);

    }

    @Test
    public void testSoftMaxCOrder() {
        NDArrays.factory().setOrder('c');
        INDArray test = NDArrays.linspace(1,4,4).reshape(2,2);
        INDArray softMaxColumns = Activations.softmax().apply(test);
        INDArray softMaxRows = Activations.softMaxRows().apply(test);
        float columnSums = softMaxColumns.sum(0).get(0);
        float rowSums = softMaxRows.sum(1).get(0);
        assertEquals(2,softMaxColumns.sum(Integer.MAX_VALUE).get(0),1e-1);
        assertEquals(2,softMaxRows.sum(Integer.MAX_VALUE).get(0),1e-1);

    }

}
