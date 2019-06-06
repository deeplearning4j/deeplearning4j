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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintStream;
import java.io.Writer;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;



/**
 * Show informations in console.<br>
 * if required save informations in file.<br>
 * 
 *
 * @author clavvis 
 */

public class SIS {
	// System Informations Saving
	//
	private String  baseModuleCode = "SIS";
	private String  moduleCode     = "?";
	//
	private PrintStream out;
	@SuppressWarnings("unused")
	private PrintStream err;
	//
	private String fullFileName = "?";
	//
	private boolean wasOpenedFile = false;
	private boolean wasClosedFile = false;
	//
	private File    sis_File;
	private Writer  sis_Writer;
	//
	private int     writerErrorInfoCount = 0;
	private int     closedFileInfoCount  = 0;
	//
	private long    charsCount = 0;
	//
	
	/**
	 * <b>initValues</b><br>
	 * public void initValues( int mtLv, String superiorModuleCode,<br>
	 *    PrintStream out, PrintStream err )<br>
	 * Initialize values for console - not file.<br>
	 * @param mtLv - method level
	 * @param superiorModuleCode - superior module code
	 * @param out - console standard output
	 * @param err - console error output (not used)
	 */
	public void initValues(
			int mtLv,
			String superiorModuleCode,
			PrintStream out,
			PrintStream err
			) {
		//
		mtLv ++;
		//
		moduleCode = superiorModuleCode + "." + baseModuleCode;
		//
		this.out = out;
		this.err = err;
		//
	}
	
	/**
	 * <b>initValues</b><br>
	 * public void initValues( int mtLv, String superiorModuleCode,<br>
	 *   PrintStream out, PrintStream err, String fileDrcS,<br>
	 *   String base_FileCode, String spc_FileCode,<br>
	 *   boolean ShowBriefInfo, boolean ShowFullInfo )<br>
	 * Initialize values for console and file.<br>
	 * 	fullFileName =<br>
	 *    "Z" +<br>
	 *	  TimeS + "_" +<br>
	 *    base_FileCode + "_" +<br>
	 *    spc_FileCode +<br>
	 *    ".txt";<br>
	 * TimeS (time string) format: "yyyyMMdd'_'HHmmss.SSS"<br>
	 * @param mtLv - method level
	 * @param superiorModuleCode - superior module code
	 * @param out - console standard output
	 * @param err - console error output (not used)
	 * @param fileDrcS - file directory as string
	 * @param base_FileCode - base part of file code
	 * @param spc_FileCode - specifying part of file code
	 * @param ShowBriefInfo - show brief informations
	 * @param ShowFullInfo - show full informations
	 */
	public void initValues(
			int mtLv,
			String superiorModuleCode,
			PrintStream out,
			PrintStream err,
			String fileDrcS,
			String base_FileCode,
			String spc_FileCode,
			boolean ShowBriefInfo,
			boolean ShowFullInfo
			) {
		//
		mtLv ++;
		//
		moduleCode = superiorModuleCode + "." + baseModuleCode;
		//
		String methodName = moduleCode + "." + "initValues";
		//
		this.out = out;
		this.err = err;
		//
		if ( ShowBriefInfo || ShowFullInfo ) {
	    	out.format( "" );
	    	out.format( BTools.getMtLvESS( mtLv ) );
			out.format( methodName + ": " );
			out.format( "fileDrcS: " + fileDrcS + "; " );
			out.format( "base_FileCode: " + base_FileCode + "; " );
			out.format( "spc_FileCode: " + spc_FileCode + "; " );
//			out.format( "STm: %s; ", Tools.getSDatePM( System.currentTimeMillis(), "HH:mm:ss" ) + "; " );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
		}
		//
		initFile( mtLv, fileDrcS, base_FileCode, spc_FileCode, ShowBriefInfo, ShowFullInfo );
		//
	}
	
	private void initFile(
			int mtLv,
			String fileDrcS,
			String base_FileCode,
			String spc_FileCode,
			boolean ShowBriefInfo,
			boolean ShowFullInfo
			) {
		//
		mtLv ++;
		//
		String oinfo = "";
		//
		String methodName = moduleCode + "." + "initFile";
		//
		if ( ShowBriefInfo || ShowFullInfo ) {
	    	out.format( "" );
	    	out.format( BTools.getMtLvESS( mtLv ) );
			out.format( methodName + ": " );
			out.format( "fileDrcS: " + fileDrcS + "; " );
			out.format( "base_FileCode: " + base_FileCode + "; " );
			out.format( "spc_FileCode: " + spc_FileCode + "; " );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
		}
		//
		spc_FileCode = spc_FileCode.replace( ":", "" );
		spc_FileCode = spc_FileCode.replace( "/", "" );
		spc_FileCode = spc_FileCode.replace( ".", "" );
		//
		File fileDrc  = new File( fileDrcS );
		//
		if ( !fileDrc.exists() ) {
			fileDrc.mkdirs();
			//
			out.format( "" );
			out.format( BTools.getMtLvESS( mtLv ) );
			out.format( methodName + ": " );
			out.format( "fileDrcS: %s; ", fileDrcS );
			out.format( "Directory was created; " );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
		}
		//
    	LocalDateTime LDT = LocalDateTime.now();
		//
    	String TimeS = LDT.format( DateTimeFormatter.ofPattern( "yyyyMMdd'_'HHmmss.SSS" ) );
		//
		fullFileName =
			"Z" +
			TimeS + "_" +
			base_FileCode +
			"_" +
			spc_FileCode +
			".txt";
		//
		sis_File = new File( fileDrcS, fullFileName );
		//
		sis_File.setReadable( true );
		//
		if ( sis_File.exists() ) {
			if ( ShowBriefInfo || ShowFullInfo ) {
		    	out.format( "" );
		    	out.format( BTools.getMtLvESS( mtLv ) );
		    	out.format( BTools.getMtLvISS() );
		    	out.format( "delete File; " );
				out.format( "%s", BTools.getSLcDtTm() );
				out.format( "%n" );
			}
			sis_File.delete();
		}
		//
	    try {
	    	sis_File.createNewFile();
    	}
    	catch ( Exception Exc ) {
		//	Exc.printStackTrace( Err_PS );
	    	out.format( "===" );
			out.format( methodName + ": " );
			out.format( "create New File error !!! " );
			out.format( "Exception: %s; ", Exc.getMessage() );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
	    	out.format( "===" );
	    	out.format( BTools.getMtLvISS() );
			out.format( "fileDrcS: " + fileDrcS + "; " );
			out.format( "fullFileName: " + fullFileName + "; " );
			out.format( "%n" );
	    	//
			return;
	    }
	    //
	    if ( ShowFullInfo ) {
	    	out.format( "" );
	    	out.format( BTools.getMtLvESS( mtLv ) );
	    	out.format( BTools.getMtLvISS() );
			out.format( "fullFileName: " + fullFileName + "; " );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
	    }
		//
	    try {
	    	sis_Writer = new BufferedWriter( new FileWriter( sis_File ) );
    	}
    	catch ( Exception Exc ) {
	    	out.format( "===" );
			out.format( methodName + ": " );
			out.format( "create New Writer: " );
			out.format( "Exception: %s; ", Exc.getMessage() );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
	    	//
    		return ;
	    }
		//
	    wasOpenedFile = true;
	    //
	    if ( ShowFullInfo ) {
			oinfo = "";
			oinfo += BTools.getMtLvESS( mtLv );
			oinfo += methodName + ": ";
			oinfo += "fullFileName: " + fullFileName + "; ";
			out.format( "%s", BTools.getSLcDtTm() );
			info( oinfo );
	    }
	    //
	}
	
	/**
	 * <b>getfullFileName</b><br>
	 * public String getfullFileName()<br>
	 * Returns full file name<br>
	 * @return full file name
	 */
	public String getfullFileName() {
		//
		return fullFileName;
	}
	
	/**
	 * <b>info</b><br>
	 * public void info( String oinfo )<br>
	 * This method is input for informations.<br>
	 * Informations are showed in console and saved in file.<br>
	 * @param oinfo - information
	 */
	public void info( String oinfo ) {
		//
		String methodName = moduleCode + "." + "info";
		//
		out.format( "%s%n", oinfo );
		//
		charsCount += oinfo.length();
		//
		String FOInfo = getFullInfoString( oinfo );
		//
		if ( !isFileOpen( methodName ) ) return;
		//
		outFile( FOInfo );
		//
        flushFile();
		//
	}
	
	/**
	 * <b>getcharsCount</b><br>
	 * public long getcharsCount()<br>
	 * Returns chars count counted from SIS creating.<br>
	 * @return chars count
	 */
	public long getcharsCount() {
		//
		return charsCount;
	}
	
	private String getFullInfoString( String oinfo ) {
		//
		String Result = "";
		//
    	LocalDateTime LDT = LocalDateTime.now();
    	//
    	String TimeS = LDT.format( DateTimeFormatter.ofPattern( "yyyy.MM.dd HH:mm:ss.SSS" ) );
		//
		Result =
			TimeS +
		 	": " +
			oinfo +
			"\r\n" +
			"";
		//
		return Result;
	}
	
	private boolean isFileOpen( String SourceMethodName ) {
		//
		if ( !wasOpenedFile ) return false; 
		if ( !wasClosedFile ) return true; 
		//
		String methodName = moduleCode + "." + "isFileOpen";
		//
		closedFileInfoCount ++;
		if ( closedFileInfoCount <= 3 ) {
	    	out.format( "===" );
//			out.format( methodName + ": " );
			out.format( methodName + "(from " + SourceMethodName + "): " );
	    	out.format( "File is closed !!!; " );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
		}
		//
		return false;
	}
	
	private void outFile( String FOInfo ) {
		//
		String methodName = moduleCode + "." + "outFile";
		//
        try {
        	sis_Writer.write( FOInfo );
        }
        catch ( Exception Exc ) {
    		if ( writerErrorInfoCount < 2 ) {
    			writerErrorInfoCount ++;
        		out.format( "===" );
    			out.format( methodName + ": " );
    			out.format( "Writer.write error !!!; " );
    			out.format( "Exception: %s; ", Exc.getMessage() );
    			out.format( "%s", BTools.getSLcDtTm() );
    			out.format( "%n" );
    		}
			//
        }
		//
	}
	
	private void flushFile() {
		//
		String methodName = moduleCode + "." + "flushFile";
		//
		try {
			sis_Writer.flush();
		}
		catch ( Exception Exc ) {
	    	out.format( "===" );
			out.format( methodName + ": " );
			out.format( "Writer.flush error !!!; " );
			out.format( "Exception: %s; ", Exc.getMessage() );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
		}
		//
	}
	
	/**
	 * <b>onStop</b><br>
	 * public void onStop( int mtLv )<br>
	 * This method should be called at the end of program.<br>
	 * Close file.<br>
	 * @param mtLv - method level
	 */
	public void onStop( int mtLv ) {
		//
		mtLv ++;
		//
		String oinfo = "";
		//
		String methodName = moduleCode + "." + "onStop";
		//
		oinfo = "";
		oinfo += BTools.getMtLvESS( mtLv );
		oinfo += methodName + ": ";
		oinfo += BTools.getSLcDtTm();
		info( oinfo );
		//
		closeFile();
		//
	}
	
	
	private void closeFile() {
		//
		String methodName = moduleCode + "." + "closeFile";
		//
		flushFile();
		//
		try {
			sis_Writer.close();
		}
		catch ( Exception Exc ) {
	    	out.format( "===" );
			out.format( methodName + ": " );
			out.format( "Writer.close error !!!; " );
			out.format( "Exception: %s; ", Exc.getMessage() );
			out.format( "%s", BTools.getSLcDtTm() );
			out.format( "%n" );
		}
		//
		wasClosedFile = true;
		//
	}
	
	
	
	
	
	
}