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

    java.util.Random rng = new java.util.Random(1230) ;

    @Test
    public void testSgetrf1() throws Exception {
	int m = 3 ;
	int n = 3 ;
	INDArray arr = Nd4j.create( new float[]{ 
			1.f, 4.f,  7.f,
			2.f, 5.f, -2.f,
			3.f, 0.f,  3.f }, 
			new int[] { m, n }, 'f' ) ;

        INDArray INFO = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createInt(1), 
        		Nd4j.getShapeInfoProvider().createShapeInformation(new int[]{1, 1}));

        int mn = Math.min( m, n ) ;
        INDArray IPIV = Nd4j.createArrayFromShapeBuffer(Nd4j.getDataBufferFactory().createInt(mn), 
        		Nd4j.getShapeInfoProvider().createShapeInformation(new int[]{1, mn}));

        Nd4j.getBlasWrapper().lapack().getrf( m, n, arr, m, IPIV, INFO);
	
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

    public void testGetrf() throws Exception {
	m = 150 ;
	n = 100 ;
	float f[] = new float[ m * n ] ;
	for( int i=0 ; i<f.length ; i++ ) 
		f[i] = rng.nextInt( 5 ) + 1 ;
 	// there is a very very small (non-zero) chance that the random matrix is singular
	// and may fail a test
        long start = System.currentTimeMillis() ;
        
        INDArray IPIV = null ;
        final int N = 100 ;
        for( int i=0 ; i<N ; i++ ) {
            arr = Nd4j.create(f, new int[] { m, n }, 'f' ) ;
        	IPIV = Nd4j.getBlasWrapper().lapack().getrf( arr );
        }
        
        INDArray L = Nd4j.getBlasWrapper().lapack().getLFactor(arr) ;
        INDArray U = Nd4j.getBlasWrapper().lapack().getUFactor(arr) ;
        INDArray P = Nd4j.getBlasWrapper().lapack().getPFactor(m, IPIV) ;

        INDArray orig = P.mmul( L ).mmul( U ) ;
    
	assertEquals( "PxLxU is not the expected size - rows", orig.size(0), arr.size(0) ) ;
	assertEquals( "PxLxU is not the expected size - cols", orig.size(1), arr.size(1) ) ;

        arr = Nd4j.create(f, new int[] { m, n }, 'f' ) ;
        for( int r=0 ; r<orig.size(0) ; r++ ) {
            for( int c=0 ; c<orig.size(1) ; c++ ) {
		assertEquals( "Original & recombined matrices differ", orig.getFloat(r,c), arr.getFloat(r,c) ), 0.001f ) ;
            }
        }
    }
}
