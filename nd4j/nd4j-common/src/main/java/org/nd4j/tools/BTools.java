/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.tools;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;

/**
 * includes several base tools
 *
 * 
 *
 * @author clavvis 
 */

//B = Base
public class BTools {
	//
	
	/**
	 * <b>getMtLvESS</b><br>
	 * public static String getMtLvESS( int mtLv )<br>
	 * Returns string. String length create indentation(shift) of other text.<br>
	 * Indentation depends on method level - great method level, great indentation.<br>
	 * Main method has method level 0.<br>
	 * Other called method has method level 1, 2,...N.<br> 
	 * @param mtLv - method level
	 * @return method level external shift string
	 */
	public static String getMtLvESS( int mtLv ) {
		//  MtLvESS = Method Level External Shift String 
		//
		if ( mtLv < 0 ) return "?";
		//
		String Result = "";
		//
	//	String LvS = ". ";
		String LvS = ".";
		//
		for ( int K = 1; K <= mtLv; K ++ ) {
			//
			Result = Result + LvS;
		}
		//
		return Result;
	}
	
	/**
	 * <b>getMtLvISS</b><br>
	 * public static String getMtLvISS()<br>
	 * Returns string. String create indentation(shift)<br>
	 *   internal text to start text of method.<br>
	 * 
	 * @return method level internal shift string
	 */
	public static String getMtLvISS() {
		//  MtLvISS = Method Level Intern Shift String 
		//
	//	String Result = "..";
	//	String Result = "~";
		String Result = " ";
		//
		return Result;
	}
	
	/**
	 * <b>getSpaces</b><br>
	 * public static String getSpaces( int SpacesCount )<br>
	 * Returns asked count of spaces.<br>
	 * If count of spaces is < 0 returns '?'. 
	 * @param SpacesCount = spaces count
	 * @return spaces
	 */
	public static String getSpaces( int SpacesCount ) {
		//
		if ( SpacesCount < 0 ) return "?";
		//
		String Info = "";
		//
		for ( int K = 1; K <= SpacesCount; K ++ ) {
			Info += " ";
		}
		//
		//
		return Info;
	}
	
	/**
	 * <b>getSBln</b><br>
	 * public static String getSBln( boolean... blnA )<br>
	 * Returns boolean(s) converted to char (true = 'T'; false = 'F')<br>
	 * If blnA.length is > 1 returns chars without separator.<br>
	 * If blnA is '{ true, false, true }' returns 'TFT'.<br>
	 * If blnA is null returns '?'.<br>
	 * If blnA.length is 0 returns '?'.<br>
	 * @param blnA
	 * @return boolean(s) as string
	 */
	public static String getSBln( boolean... blnA ) {
		//
		String Info = "";
		//
		if ( blnA == null ) return "?";
		if ( blnA.length == 0 ) return "?";
		//
		for ( int K = 0; K < blnA.length; K ++ ) {
			//
			Info += ( blnA[ K ] )? "T" : "F";
		}
		//
		return Info;
	}
		
	/**
	 * <b>getSDbl</b><br>
	 * public static String getSDbl( double Value, int DecPrec )<br>
	 * Returns double converted to string.<br>
	 * If Value is Double.NaN returns "NaN".<br>
	 * If DecPrec is < 0 is DecPrec set 0.<br>
	 * 
	 * @param Value - value
	 * @param DecPrec - decimal precision
	 * @return double as string
	 */
	public static String getSDbl( double Value, int DecPrec ) {
		//
		String Result = "";
		//
		if ( Double.isNaN( Value ) ) return "NaN";
		//
		if ( DecPrec < 0 ) DecPrec = 0;
		//
		String DFS = "###,###,##0";
		//
		if ( DecPrec > 0 ) {
			int idx = 0;
			DFS += ".";
			while ( idx < DecPrec ) {
				DFS = DFS + "0";
				idx ++;
				if ( idx > 100 ) break;
			}
		}
		//
//		Locale locale  = new Locale("en", "UK");
		//
		DecimalFormatSymbols DcmFrmSmb = new DecimalFormatSymbols( Locale.getDefault());
		DcmFrmSmb.setDecimalSeparator('.');
		DcmFrmSmb.setGroupingSeparator(' ');
		//
		DecimalFormat DcmFrm;
		//
		DcmFrm = new DecimalFormat( DFS, DcmFrmSmb );
		//
	//	DcmFrm.setGroupingSize( 3 );
		//
		Result = DcmFrm.format( Value );
		//
		return Result;
	}
	
	/**
	 * <b>getSDbl</b><br>
	 * public static String getSDbl( double Value, int DecPrec, boolean ShowPlusSign )<br>
	 * Returns double converted to string.<br>
	 * If Value is Double.NaN returns "NaN".<br>
	 * If DecPrec is < 0 is DecPrec set 0.<br>
	 * If ShowPlusSign is true:<br>
	 *   - If Value is > 0 sign is '+'.<br>
	 *   - If Value is 0 sign is ' '.<br>
	 * @param Value - value
	 * @param DecPrec - decimal precision
	 * @param ShowPlusSign - show plus sign
	 * @return double as string
	 */
	public static String getSDbl( double Value, int DecPrec, boolean ShowPlusSign ) {
		//
		String PlusSign = "";
		//
		if ( ShowPlusSign && Value  > 0 ) PlusSign = "+";
		if ( ShowPlusSign && Value == 0 ) PlusSign = " ";
		//
		return PlusSign + getSDbl( Value, DecPrec );
	}
	
	/**
	 * <b>getSDbl</b><br>
	 * public static String getSDbl( double Value, int DecPrec, boolean ShowPlusSign, int StringLength )<br>
	 * Returns double converted to string.<br>
	 * If Value is Double.NaN returns "NaN".<br>
	 * If DecPrec is < 0 is DecPrec set 0.<br>
	 * If ShowPlusSign is true:<br>
	 *   - If Value is > 0 sign is '+'.<br>
	 *   - If Value is 0 sign is ' '.<br>
	 * If StringLength is > base double string length<br>
	 *   before base double string adds relevant spaces.<br>
	 * If StringLength is <= base double string length<br>
	 *   returns base double string.<br>
	 * @param Value - value
	 * @param DecPrec - decimal precision
	 * @param ShowPlusSign - show plus sign
	 * @param StringLength - string length
	 * @return double as string
	 */
	public static String getSDbl( double Value, int DecPrec, boolean ShowPlusSign, int StringLength ) {
		//
		String Info = "";
		//
		String SDbl = getSDbl( Value, DecPrec, ShowPlusSign );
		//
		if ( SDbl.length() >= StringLength ) return SDbl;
		//
//		String SpacesS = "            ";
		String SpacesS = getSpaces( StringLength );
		//
		Info = SpacesS.substring( 0, StringLength - SDbl.length() ) + SDbl;
		//
		return Info;
	}
	
	/**
	 * <b>getSInt</b><br>
	 * public static String getSInt( int Value, int CharsCount )<br>
	 * Returns int converted to string.<br>
	 * If CharsCount > base int string length<br>
	 *   before base int string adds relevant spaces.<br>
	 * If CharsCount <= base int string length<br>
	 *   returns base int string.<br>
	 * @param Value - value
	 * @param CharsCount - chars count
	 * @return int as string
	 */
	public static String getSInt( int Value, int CharsCount ) {
		//
		return getSInt( Value, CharsCount, ' ' );
	}
	
	/**
	 * <b>getSInt</b><br>
	 * public static String getSInt( int Value, int CharsCount, char LeadingChar )<br>
	 * Returns int converted to string.<br>
	 * If CharsCount > base int string length<br>
	 *   before base int string adds relevant leading chars.<br>
	 * If CharsCount <= base int string length<br>
	 *   returns base int string.<br>
	 * 
	 * @param Value - value
	 * @param CharsCount - chars count
	 * @param LeadingChar - leading char
	 * @return int as string
	 */
	public static String getSInt( int Value, int CharsCount, char LeadingChar ) {
		//
		String Result = "";
		//
		if ( CharsCount <= 0 ) {
			return getSInt( Value );
		}
		//
		String FormatS = "";
		if ( LeadingChar == '0' ) {
			FormatS = "%" + LeadingChar + Integer.toString( CharsCount ) + "d";
		}
		else {
			FormatS = "%" + Integer.toString( CharsCount ) + "d";
		}
		//
		Result = String.format( FormatS, Value );
		//
		return Result;
	}
	
	/**
	 * <b>getSInt</b><br>
	 * public static String getSInt( int Value )<br>
	 * Returns int converted to string.<br>
	 * @param Value
	 * @return int as string
	 */
	public static String getSInt( int Value ) {
		//
		String Result = "";
		//
		Result = String.format( "%d", Value );
		//
		return Result;
	}
	
	/**
	 * <b>getSIntA</b><br>
	 * public static String getSIntA( int... intA )<br>
	 * Returns intA converted to string.<br>
	 * Strings are separated with ", ".<br>
	 * If intA is null returns '?'.<br>
	 * If intA.length is 0 returns '?'.<br>
	 * @param intA - int value(s) (one or more)
	 * @return int... as string
	 */
//	public static String getSIntA( int[] intA ) {
	public static String getSIntA( int... intA ) {
		//
		String Info = "";
		//
		if ( intA == null ) return "?";
		if ( intA.length == 0 ) return "?";
		//
		for ( int K = 0; K < intA.length; K ++ ) {
			//
            Info += ( Info.isEmpty() )? "" : ", ";
			Info += BTools.getSInt( intA[ K ] );
		}
		//
		return Info;
	}
	
	/**
	 * <b>getIndexCharsCount</b><br>
	 * public static int getIndexCharsCount( int MaxIndex )<br>
	 * Returns chars count for max value of index.<br>
	 * Example: Max value of index is 150 and chars count is 3.<br>
	 * It is important for statement of indexed values.<br>
	 * Index columns can have the same width for all rouws.<br> 
	 * @param MaxIndex - max value of index
	 * @return chars count for max value of index
	 */
	public static int getIndexCharsCount( int MaxIndex ) {
		//
		int CharsCount = 1;
		//
		if ( MaxIndex <= 0 ) return 1;
		//
		CharsCount = (int)Math.log10( MaxIndex ) + 1;
		//
		return CharsCount;
	}
	
	/**
	 * <b>getSLcDtTm</b><br>
	 * public static String getSLcDtTm()<br>
	 * Returns local datetime as string.<br>
	 * Datetime format is "mm:ss.SSS".<br>
	 * @return local datetime as string
	 */
	public static String getSLcDtTm() {
		//
		return getSLcDtTm( "mm:ss.SSS" );
	}
	
	/**
	 * <b>getSLcDtTm</b><br>
	 * public static String getSLcDtTm( String FormatS )<br>
	 * Returns local datetime as string.<br>
	 * Datetime format is param.<br>
	 * @param FormatS datetime format
	 * @return local datetime as string
	 */
	public static String getSLcDtTm( String FormatS ) {
		//
		String Result = "?";
		//
    	LocalDateTime LDT = LocalDateTime.now();
    	//
    	Result = "LDTm: " +  LDT.format( DateTimeFormatter.ofPattern( FormatS ) );
    	//
		return Result;
	}
	
	
	
	
	
	
}