package org.deeplearning4j.iterativereduce.runtime.io;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class TestSVMLightRecordFactory {
	
	private String test_svm_light_w_comment = "-1 1:0.43 3:0.12 9284:0.2 # abcdef";
	private String test_svm_light_no_comment = "-1 1:0.43 3:0.12 9284:0.2";
	private String test_svm_light_positive_label = "1 1:0.43 3:0.12 9284:0.2";
	private String test_svm_light_no_label = "1:0.43 3:0.12 9284:0.2";
	
	

	@Test
	public void testSVMLightRecordFactoryParseWithComment() {
		
		int feature_size = 9285;
		
		INDArray in_vector = Nd4j.create( feature_size );
		INDArray out_vector = Nd4j.create( 1 );
		
		SVMLightRecordFactory recFactory = new SVMLightRecordFactory( feature_size );
		recFactory.parseFromLine( test_svm_light_w_comment, in_vector, out_vector );
		
		assertEquals( -1.0, out_vector.getDouble( 0 ), 0.0 );
		
		
	}

	@Test
	public void testSVMLightRecordFactoryParseWithNoComment() {
		
		int feature_size = 9285;
		
		INDArray in_vector = Nd4j.create( feature_size );
		INDArray out_vector = Nd4j.create( 1 );
		
		SVMLightRecordFactory recFactory = new SVMLightRecordFactory( feature_size );
		recFactory.parseFromLine( test_svm_light_no_comment, in_vector, out_vector );
		
		assertEquals( -1.0, out_vector.getDouble( 0 ), 0.0 );
		
		
	}

	@Test
	public void testSVMLightRecordFactoryParseWithPositiveLabel() {
		
		int feature_size = 9285;
		
		INDArray in_vector = Nd4j.create( feature_size );
		INDArray out_vector = Nd4j.create( 1 );
		
		SVMLightRecordFactory recFactory = new SVMLightRecordFactory( feature_size );
		recFactory.parseFromLine( test_svm_light_positive_label, in_vector, out_vector );
		
		assertEquals( 1.0, out_vector.getDouble( 0 ), 0.0 );
		
		
	}
	
	@Test
	public void testSVMLightRecordFactoryParseNoLabelException() {
		
		int feature_size = 9285;
		boolean caughtParseException = false;
		
		INDArray in_vector = Nd4j.create( feature_size );
		INDArray out_vector = Nd4j.create( 1 );
		
		SVMLightRecordFactory recFactory = new SVMLightRecordFactory( feature_size );
		try {
			recFactory.parseFromLine( test_svm_light_no_label, in_vector, out_vector );
		} catch (NumberFormatException e ) {
			// should catch this
			caughtParseException = true;
		}
		
		assertEquals( true, caughtParseException );
		
		
	}	
	
	
}
