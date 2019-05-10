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

import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 *
 * 
 *
 * @author clavvis
 */

public class InfoLineTest {
	//
	
	@Test
	public void testAll() throws Exception {
		//
		InfoValues iv0 = new InfoValues( " A", " B" );
		InfoValues iv1 = new InfoValues( " C", " D" );
		InfoValues iv2 = new InfoValues( " E", " F", " G", " H" );
		//
		iv0.vsL.add( " ab " );
		iv1.vsL.add( " cd " );
		iv2.vsL.add( " ef " );
		//
		InfoLine il = new InfoLine();
		//
		il.ivL.add( iv0 );
		il.ivL.add( iv1 );
		il.ivL.add( iv2 );
		//
		int mtLv = 2;
		//
		assertEquals( ".. | A  | C  | E  |", il.getTitleLine( mtLv, 0 ) );
		assertEquals( ".. | B  | D  | F  |", il.getTitleLine( mtLv, 1 ) );
		assertEquals( ".. |    |    | G  |", il.getTitleLine( mtLv, 2 ) );
		assertEquals( ".. |    |    | H  |", il.getTitleLine( mtLv, 3 ) );
		assertEquals( ".. | ab | cd | ef |", il.getValuesLine( mtLv ) );
		//
	}
	
	
	
}