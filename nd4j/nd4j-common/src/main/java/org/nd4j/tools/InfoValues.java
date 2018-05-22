package org.nd4j.tools;

import java.util.ArrayList;
import java.util.List;

/**
 * Save value and it's titles for one column.<br>
 * Titles strings in array create one title column.<br>
 * One main column can have several sub columns.<br>
 * Columns are separated with char '|'.<br>
 * Collaborate with class InfoLine.<br>
 * @author clavvis 
 */

public class InfoValues {
	//
	public InfoValues( String... titleA ) {
		//
		for ( int i = 0; i < this.titleA.length; i++ ) this.titleA[ i ] = "";
		//
		int Max_K = Math.min( this.titleA.length - 1, titleA.length - 1 );
		//
		for ( int i = 0; i <= Max_K; i++ ) this.titleA[ i ] = titleA[ i ];
		//
	}
	//
	/**
	 * Title array.<br>
	 */
	public String[] titleA = new String[ 6 ];
	//
	// VS = Values String
	/**
	 * Values string list.<br>
	 */
	public List< String > vsL = new ArrayList< String >();
	//
	
	/**
	 * Returns values.<br>
	 * This method use class InfoLine.<br>
	 * This method is not intended for external use.<br>
	 * @return
	 */
	public String getValues() {
		//
		String info = "";
		//
		for ( int i = 0; i < vsL.size(); i ++ ) {
			//
			info += vsL.get( i ) + "|";
		}
		//
		return info;
	}
	
}