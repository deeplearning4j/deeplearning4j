package org.deeplearning4j.tools2;

import org.deeplearning4j.tools.SIS;
import org.deeplearning4j.tools.DL4JTools;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;


import org.junit.After;
import org.junit.Test;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;

import static org.junit.Assert.*;

/**
 *
 *
 * 
 *
 * @author clavvis
 */

public class DL4JToolsTest {
	//
	@Rule
	public TemporaryFolder tmpFld = new TemporaryFolder();
	//
	private SIS sis;
	//
	@Test
	public void test_all() throws Exception {
		//
		sis = new SIS();
		//
		int mtLv = 0;
		//
		sis.initValues( mtLv, "TEST", System.out, System.err, tmpFld.getRoot().getAbsolutePath(), "Test", "ABC", true, true );
		//
		INDArray in_INDA = Nd4j.zeros( 8, 8 );
		INDArray ot_INDA = Nd4j.ones( 8, 1 );
		//
		ot_INDA.putScalar( 7, 5 );
		//
		DataSet ds = new DataSet( in_INDA, ot_INDA );
		//
		DL4JTools dl4jt = new DL4JTools( sis, "TEST" );
		//
		dl4jt.showDataSet( mtLv, "ds", ds, 2, 2, 20, 20 );
		//
	//	assertEquals( 100, sis.getcharsCount() );
		//
		assertTrue( sis.getcharsCount() > 1190 && sis.getcharsCount() < 1210 );
		//
		INDArray spec_INDA = Nd4j.zeros( 8, 8 );
		//
		dl4jt.showINDArray( mtLv, "spec_INDA", spec_INDA, 3, 20, 20 );
		//
	//	assertEquals( 100, sis.getcharsCount() );
		//
		assertTrue( sis.getcharsCount() > 2150 && sis.getcharsCount() < 2160 );
		//
		//
	}
	
	@After
	public void after() {
		//
		int mtLv = 0;
		if ( sis != null ) sis.onStop( mtLv );
		//
	//	tmpFld.delete();
		//
	}
	
	
	
}