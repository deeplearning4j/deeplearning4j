package org.deeplearning4j.linalg.factory;


import static org.junit.Assert.*;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.junit.Test;

/**
 * @author Adam Gibson
 */
public abstract class BlasWrapperTest {

    @Test
    public void testIaMax() {
         INDArray test = NDArrays.create(new float[]{1,2,3,4});
         int max = NDArrays.getBlasWrapper().iamax(test);
         assertEquals(3,max);


        INDArray rows = NDArrays.create(new float[]{1,3,2,4},new int[]{2,2});
        for(int i = 0; i < rows.rows(); i++) {
            INDArray row = rows.getRow(i);
            int max2 = NDArrays.getBlasWrapper().iamax(row);
            assertEquals(1,max2);
        }

    }




}
