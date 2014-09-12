package org.nd4j.linalg.factory;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import static org.junit.Assert.*;


/**
 * Created by agibsonccc on 9/11/14.
 */
public abstract class BlasWrapperTests {

    @Test
    public void axpyTest() {
        INDArray a = Nd4j.getBlasWrapper().axpy(1,Nd4j.ones(3),Nd4j.ones(3));
        INDArray a2 = Nd4j.create(new float[]{2,2,2});
        assertEquals(a2,a);

        INDArray matrix = Nd4j.linspace(1,4,4).reshape(2,2);
        INDArray row = matrix.getRow(1);
        INDArray result = Nd4j.create(new float[]{1,2});
        Nd4j.getBlasWrapper().axpy(1,row,result);
        assertEquals(Nd4j.create(new float[]{3,6}),result);

    }


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
