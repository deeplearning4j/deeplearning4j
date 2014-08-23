package org.deeplearning4j.linalg.jblas;


import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import static org.junit.Assert.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



/**
 * NDArrayTests
 * @author Adam Gibson
 */
public class NDArrayTests extends org.deeplearning4j.linalg.api.test.NDArrayTests {
    private static Logger log = LoggerFactory.getLogger(NDArrayTests.class);


    @Test
    public void testResults() {
        double[][] data = new double[][]{
                {1,2,3,4},
                {5,6,7,8}
        };

        double[] mmul = {1,2,3,4};

        DoubleMatrix d = new DoubleMatrix(data);
        INDArray d2 = NDArrays.create(data);
        assertEquals(d.rows,d2.rows());
        assertEquals(d.columns,d2.columns());
        verifyElements(d,d2);

        INDArray toMmulD2 = NDArrays.create(mmul);
        DoubleMatrix toMmulD = new DoubleMatrix(mmul);

        DoubleMatrix mmulResultD = d.mmul(toMmulD);

        INDArray mmulResultD2 = d2.mmul(toMmulD2);




    }


    protected void verifyElements(DoubleMatrix d,INDArray d2) {
        for(int i = 0; i < d.rows; i++) {
            for(int j = 0; j < d.columns; j++) {
                double test1 = d.get(i,j);
                double test2 = (double) d2.getScalar(i,j).element();
                assertEquals(test1,test2,1e-6);
            }
        }
    }

}
