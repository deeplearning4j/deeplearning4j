package org.nd4j.tools;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 *
 * 
 *
 * @author clavvis 
 */

public class InfoValuesTest {
	//
	private String[] t1_titleA = { "T0", "T1", "T2", "T3", "T4", "T5" };
	//
	private String[] t2_titleA = { "", "T1", "T2" };
	//
	
	@Test
	public void testconstructor() throws Exception {
		//
		InfoValues iv;
		//
		iv = new InfoValues( t1_titleA );
		assertEquals( "T0", iv.titleA[ 0 ] );
		assertEquals( "T1", iv.titleA[ 1 ] );
		assertEquals( "T2", iv.titleA[ 2 ] );
		assertEquals( "T3", iv.titleA[ 3 ] );
		assertEquals( "T4", iv.titleA[ 4 ] );
		assertEquals( "T5", iv.titleA[ 5 ] );
		//
		iv = new InfoValues( t2_titleA );
		assertEquals( "", iv.titleA[ 0 ] );
		assertEquals( "T1", iv.titleA[ 1 ] );
		assertEquals( "T2", iv.titleA[ 2 ] );
		assertEquals( "", iv.titleA[ 3 ] );
		assertEquals( "", iv.titleA[ 4 ] );
		assertEquals( "", iv.titleA[ 5 ] );
		//
	}
	
	@Test
	public void testgetValues() throws Exception {
		//
		InfoValues iv;
		//
		iv = new InfoValues( "Test" );
		iv.vsL.add( " AB " );
		iv.vsL.add( " CD " );
		//
		assertEquals( " AB | CD |", iv.getValues() );
		//
	}
	
	
}
