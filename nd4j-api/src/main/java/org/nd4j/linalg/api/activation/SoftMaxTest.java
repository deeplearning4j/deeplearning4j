package org.nd4j.linalg.api.activation;

import static org.junit.Assert.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDArrays;
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
        INDArray test = NDArrays.linspace(1,6,6).reshape(2,3);
        INDArray softMaxColumns = Activations.softmax().apply(test);
        INDArray softMaxRows = Activations.softMaxRows().apply(test);
        INDArray columns = softMaxColumns.sum(0);
        INDArray rows = softMaxRows.sum(1);
        //softmax along columns: should be 1 in every cell ( note that there are 3 columns)
        assertEquals(3,columns.sum(Integer.MAX_VALUE).get(0),1e-1);
        //softmax along rows: should be 1 in every cell (note that there are 2 rows
        assertEquals(2,rows.sum(Integer.MAX_VALUE).get(0),1e-1);

    }

    @Test
    public void testSoftMaxCOrder() {
        NDArrays.factory().setOrder('c');
        INDArray test = NDArrays.linspace(1,6,6).reshape(2,3);
        INDArray softMaxColumns = Activations.softmax().apply(test);
        INDArray softMaxRows = Activations.softMaxRows().apply(test);

        INDArray columns = softMaxColumns.sum(0);
        INDArray rows = softMaxRows.sum(1);
        //softmax along columns: should be 1 in every cell ( note that there are 3 columns)
        assertEquals(3,columns.sum(Integer.MAX_VALUE).get(0),1e-1);
        //softmax along rows: should be 1 in every cell (note that there are 2 rows
        assertEquals(2,rows.sum(Integer.MAX_VALUE).get(0),1e-1);

    }

}
