package jcuda.jcublas.ops;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
@Ignore
public class CublasTests {
    @Test
    public void testSgetrf1() throws Exception {
	int m = 3 ;
	int n = 3 ;
	INDArray arr = Nd4j.create( new float[]{ 
			1.f, 4.f,  7.f,
			2.f, 5.f, -2.f,
			3.f, 0.f,  3.f }, 
			new int[] { m, n }, 'f' ) ;

	int lda = Math.min(m, n) ;
        INDArray INFO = Nd4j.create(1) ;
        INDArray IPIV = Nd4j.create( lda ) ;

        Nd4j.getBlasWrapper().lapack().getrf(m,n,arr, lda,IPIV,INFO);
	
        assertEquals( "getrf returned a non-zero code",  0, INFO.getInt(0) ) ;

	// The above matrix factorizes to :
	//   7.00000  -2.00000   3.00000
	//   0.57143   6.14286  -1.71429
	//   0.14286   0.37209   3.20930
        
        assertEquals(  7.00000f, arr.getFloat(0), 0.00001f);
        assertEquals(  0.57143f, arr.getFloat(1), 0.00001f);
        assertEquals(  0.14286f, arr.getFloat(2), 0.00001f);
        assertEquals( -2.00000f, arr.getFloat(3), 0.00001f);
        assertEquals(  6.14286f, arr.getFloat(4), 0.00001f);
        assertEquals(  0.37209f, arr.getFloat(5), 0.00001f);
        assertEquals(  3.00000f, arr.getFloat(6), 0.00001f);
        assertEquals( -1.71429f, arr.getFloat(7), 0.00001f);
        assertEquals(  3.20930f, arr.getFloat(8), 0.00001f);
    }

    public void testSgetrf2() throws Exception {
	int m = 50 ;
	int n = 50 ;
	INDArray arr = Nd4j.rand( m, n ) ;
	INDArray compare = arr.dup() ;

	int lda = Math.min(m, n) ;
        INDArray INFO = Nd4j.create(1) ;
        INDArray IPIV = Nd4j.create( lda ) ;

        Nd4j.getBlasWrapper().lapack().getrf( m, n, arr, lda, IPIV, INFO );
        assertEquals( "getrf returned a non-zero code",  0, INFO.getInt(0) ) ;
	
	// Extract the L & U factors from the overwritten output
        INDArray L = arr.dup() ;
        INDArray U = arr.dup() ;
        for( int r=0 ; r<L.size(0) ; r++ ) {
            for( int c=0 ; c<L.size(1) ; c++ ) {
            	if( r>c ) {
            		U.putScalar(r, c, 0.f ) ;
            	} else if( r<c ) {
            		L.putScalar(r, c, 0.f ) ;
            	} else {
            		L.putScalar(r, c, 1.f ) ;
            	}
            }
        }
        
	// Build the permutation matrix from the IPIV output
        INDArray P = Nd4j.eye( lda ) ;
        for( int i=0 ; i<lda ; i++ ) {
        	if( IPIV.getInt(i) > (i+1) ) {
        		INDArray r1 = P.getRow(i).dup() ;
        		INDArray r2 = P.getRow( IPIV.getInt(i)-1) ;
        		P.putRow( i, r2 ) ;
        		P.putRow( IPIV.getInt(i)-1, r1 ) ;
        	}
        }
	// The combined P x L x U should be the same as the original input matrix
        INDArray orig = P.mmul( L ).mmul( U ) ;

        assertEquals( "LxUxP is not the expected size", orig.length(), compare.length() ) ;
        for( int r=0 ; r<orig.size(1) ; r++ ) {
            for( int c=0 ; c<orig.size(0) ; c++ ) {
        	assertEquals( "Original matrix and recombined factors are not the same", orig.getFloat(r,c), compare.getFloat(r,c), 0.000001f ) ;
	    }
	}
	
    }

}
