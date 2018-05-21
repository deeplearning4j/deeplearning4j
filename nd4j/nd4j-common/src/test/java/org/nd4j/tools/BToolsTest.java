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

public class BToolsTest {
	//
	
	@Test
	public void testgetMtLvESS() throws Exception {
		//
		assertEquals( "?", BTools.getMtLvESS( -5 ) );
		assertEquals( "", BTools.getMtLvESS( 0 ) );
		assertEquals( "...", BTools.getMtLvESS( 3 ) );
		//
	}
	
	@Test
	public void testgetMtLvISS() throws Exception {
		//
		assertEquals( " ", BTools.getMtLvISS() );
		//
	}
	
	@Test
	public void testgetSpaces() throws Exception {
		//
		assertEquals( "?", BTools.getSpaces( -3 ) );
		assertEquals( "", BTools.getSpaces( 0 ) );
		assertEquals( "    ", BTools.getSpaces( 4 ) );
		//
	}
	
	@Test
	public void testgetSBln() throws Exception {
		//
		assertEquals( "?", BTools.getSBln() );
		assertEquals( "?", BTools.getSBln( null ) );
		assertEquals( "T", BTools.getSBln( true ) );
		assertEquals( "F", BTools.getSBln( false ) );
		assertEquals( "TFFT", BTools.getSBln( true, false, false, true ) );
		assertEquals( "FTFFT", BTools.getSBln( false, true, false, false, true ) );
		//
	}
	
	@Test
	public void testgetSDbl() throws Exception {
		//
		assertEquals( "NaN", BTools.getSDbl( Double.NaN, 0 ) );
		assertEquals( "-6", BTools.getSDbl( -5.5D, 0 ) );
		assertEquals( "-5.50", BTools.getSDbl( -5.5D, 2 ) );
		assertEquals( "-5.30", BTools.getSDbl( -5.3D, 2 ) );
		assertEquals( "-5", BTools.getSDbl( -5.3D, 0 ) );
		assertEquals( "0.00", BTools.getSDbl( 0D, 2 ) );
		assertEquals( "0", BTools.getSDbl( 0D, 0 ) );
		assertEquals( "0.30", BTools.getSDbl( 0.3D, 2 ) );
		assertEquals( "4.50", BTools.getSDbl( 4.5D, 2 ) );
		assertEquals( "4", BTools.getSDbl( 4.5D, 0 ) );
		assertEquals( "6", BTools.getSDbl( 5.5D, 0 ) );
		assertEquals( "12 345 678", BTools.getSDbl( 12345678D, 0 ) );
		//
		assertEquals( "-456", BTools.getSDbl( -456D, 0, false ) );
		assertEquals( "-456", BTools.getSDbl( -456D, 0, true ) );
		assertEquals( "+456", BTools.getSDbl( 456D, 0, true ) );
		assertEquals( "456", BTools.getSDbl( 456D, 0, false ) );
		assertEquals( " 0", BTools.getSDbl( 0D, 0, true ) );
		assertEquals( "0", BTools.getSDbl( 0D, 0, false ) );
		//
		assertEquals( "  4.50", BTools.getSDbl( 4.5D, 2, false, 6 ) );
		assertEquals( " +4.50", BTools.getSDbl( 4.5D, 2, true, 6 ) );
		assertEquals( "   +456", BTools.getSDbl( 456D, 0, true, 7 ) );
		assertEquals( "    456", BTools.getSDbl( 456D, 0, false, 7 ) );
		//
	}
	
	@Test
	public void testgetSInt() throws Exception {
		//
		assertEquals( "23", BTools.getSInt( 23, 1 ) );
		assertEquals( "23", BTools.getSInt( 23, 2 ) );
		assertEquals( " 23", BTools.getSInt( 23, 3 ) );
		//
		assertEquals( "0000056", BTools.getSInt( 56, 7, '0' ) );
		//
	}
	
	@Test
	public void testgetSIntA() throws Exception {
		//
		assertEquals( "?", BTools.getSIntA( null ) );
		assertEquals( "?", BTools.getSIntA(  ) );
		assertEquals( "0", BTools.getSIntA( 0 ) );
		assertEquals( "5, 6, 7", BTools.getSIntA( 5, 6, 7 ) );
		int[] intA = { 2, 3, 4, 5, 6 };
		assertEquals( "2, 3, 4, 5, 6", BTools.getSIntA( intA ) );
		//
	}
	
	@Test
	public void testgetIndexCharsCount() throws Exception {
		//
		assertEquals( 1, BTools.getIndexCharsCount( -5 ) );
		assertEquals( 1, BTools.getIndexCharsCount( 5 ) );
		assertEquals( 3, BTools.getIndexCharsCount( 345 ) );
		//
	}
	
	@Test
	public void testgetSLcDtTm() throws Exception {
		//
		assertEquals( 15, BTools.getSLcDtTm().length() );
		assertEquals( "LDTm: ", BTools.getSLcDtTm().substring( 0, 6 ) );
		//
	}
	
	
}