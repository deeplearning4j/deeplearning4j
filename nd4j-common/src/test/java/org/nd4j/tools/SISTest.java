package org.nd4j.tools;

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

public class SISTest {
	//
	@Rule
	public TemporaryFolder tmpFld = new TemporaryFolder();
	//
	private SIS sis;
	//
	
	@Test
	public void testAll() throws Exception {
		//
		sis = new SIS();
		//
		int mtLv = 0;
		//
		sis.initValues( mtLv, "TEST", System.out, System.err, tmpFld.getRoot().getAbsolutePath(), "Test", "ABC", true, true );
		//
		String fFName = sis.getfullFileName();
		sis.info( fFName );
		sis.info( "aaabbbcccdddeefff" );
		//
		assertEquals( 33, fFName.length() );
		assertEquals( "Z", fFName.substring( 0, 1 ) );
		assertEquals( "_Test_ABC.txt", fFName.substring( fFName.length() - 13, fFName.length() ) );
	//	assertEquals( "", fFName );
	//	assertEquals( "", tmpFld.getRoot().getAbsolutePath() );
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