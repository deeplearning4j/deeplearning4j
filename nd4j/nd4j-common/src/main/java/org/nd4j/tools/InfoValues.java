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