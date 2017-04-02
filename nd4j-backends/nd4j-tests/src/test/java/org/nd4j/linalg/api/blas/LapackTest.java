package org.nd4j.linalg.api.blas;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.Assert.assertEquals;

/**
 * @author rcorbish
 */
@RunWith(Parameterized.class)
public class LapackTest extends BaseNd4jTest {
    public LapackTest(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testLU() {
        INDArray A = Nd4j.create( new float[]  { 1,2,3,4,5,6,7,8,9 } );

        Nd4j.getBlasWrapper().lapack().getrf( A ) ;
    }


    @Test
    public void testQR() {
        INDArray A = Nd4j.create( new float[]  { 1,2,3,4,5,6,7,8,9 } );

        Nd4j.getBlasWrapper().lapack().geqrf( A, null ) ;
    }


    @Test
    public void testCholesky() {
        INDArray A = Nd4j.create( new float[]  { 1,2,3,4,5,6,7,8,9 } );

        Nd4j.getBlasWrapper().lapack().potrf( A, true ) ;
    }

    @Override
    public char ordering() {
        return 'f';
    }
}
