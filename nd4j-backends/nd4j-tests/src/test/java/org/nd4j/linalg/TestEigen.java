
package org.nd4j.linalg.eigen;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.eigen.Eigen ;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by rcorbish
 */
@RunWith(Parameterized.class)
public class TestEigen extends BaseNd4jTest {

    public TestEigen(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testSyev() {
	    INDArray A = Nd4j.create( new float[][] { 
	    	 { 1.96f,  -6.49f,  -0.47f,  -7.20f,  -0.65f },
		 { -6.49f,  3.80f,  -6.39f,  1.50f,  -6.34f },
		 { -0.47f, -6.39f,  4.17f,  -1.51f,  2.67f },
		 { -7.20f,  1.50f, -1.51f,  5.70f,  1.80f },
		 { -0.65f, -6.34f,  2.67f,  1.80f, -7.10f }
	    } ) ;

	    INDArray B = A.dup() ;
	    INDArray e = Eigen.symmetricGeneralizedEigenvalues(A) ;
	    
	    for( int i=0 ; i<A.rows() ; i++ ) { 
		    INDArray LHS = B.mmul( A.slice(i,1) ) ;
		    INDArray RHS = A.slice(i,1).mul( e.getFloat(i) ) ;
		    
		    for( int j=0 ; j<LHS.length() ; j++ ) {
		    	assertEquals( LHS.getFloat(j), RHS.getFloat(j), 0.001f ) ;
		    }
	    }    
	}


    @Override
    public char ordering() {
        return 'f';
    }
}

