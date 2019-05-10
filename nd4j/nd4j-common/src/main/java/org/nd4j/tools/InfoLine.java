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

import java.util.ArrayList;
import java.util.List;


/**
 * Save values and titles for one values line.<br>
 * Columns are separated with char '|'.<br>
 *
 * Collaborate with class InfoValues.<br>
 * @author clavvis 
 */

public class InfoLine {
	//
	public InfoLine() {
		//
	}
	//
	public List< InfoValues > ivL = new ArrayList< InfoValues >();
	//
	
	/**
	 * Returns titles line as string appointed by title index (0..5).<br>
	 * Columns are separated with char '|'.<br>
	 * If title index is < 0 returns "?".<br> 
	 * If title index is > 5 returns "?".<br> 
	 * @param mtLv - method level
	 * @param title_I - title index
	 * @return titles line as string
	 */
	public String getTitleLine( int mtLv, int title_I ) {
		//
		String info = "";
		//
		if ( title_I < 0 ) return "?";
		if ( title_I > 5 ) return "?";
		//
		info = "";
		info += BTools.getMtLvESS( mtLv );
		info += BTools.getMtLvISS();
		info += "|";
		//
		InfoValues i_IV;
		//
		String i_ValuesS = "";
		//
		int i_VSLen = -1;
		//
		String i_TitleS = "";
		//
		for ( int i = 0; i < ivL.size(); i ++ ) {
			//
			i_IV = ivL.get( i );
			//
			i_ValuesS = i_IV.getValues();
			//
			i_VSLen = i_ValuesS.length();
			//
			i_TitleS = ( title_I < i_IV.titleA.length )? i_IV.titleA[ title_I ] : "";
			//
			i_TitleS = i_TitleS + BTools.getSpaces( i_VSLen );
			//
			info += i_TitleS.substring( 0, i_VSLen - 1 );
			//
			info += "|";
		}
		//
		return info;
	}
	
	/**
	 * Returns values line as string.<br>
	 * Columns are separated with char '|'.<br>
	 * @param mtLv - method level
	 * @return values line as string
	 */
	public String getValuesLine( int mtLv ) {
		//
		String info = "";
		//
		info += BTools.getMtLvESS( mtLv );
		info += BTools.getMtLvISS();
		info += "|";
		//
		InfoValues i_IV;
		//
		for ( int i = 0; i < ivL.size(); i ++ ) {
			//
			i_IV = ivL.get( i );
			//
			info += i_IV.getValues();
		}
		//
		return info;
	}
	
	
		
}