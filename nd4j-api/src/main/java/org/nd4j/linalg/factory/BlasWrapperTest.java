package org.nd4j.linalg.factory;


import static org.junit.Assert.*;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.junit.Test;

/**
 * @author Adam Gibson
 */
public abstract class BlasWrapperTest {

    @Test
    public void testIaMax() {
         INDArray test = Nd4j.create(new float[]{1, 2, 3, 4});
         int max = Nd4j.getBlasWrapper().iamax(test);
         assertEquals(3,max);


        INDArray rows = Nd4j.create(new float[]{1, 3, 2, 4}, new int[]{2, 2});
        for(int i = 0; i < rows.rows(); i++) {
            INDArray row = rows.getRow(i);
            int max2 = Nd4j.getBlasWrapper().iamax(row);
            assertEquals(1,max2);
        }

    }




}
