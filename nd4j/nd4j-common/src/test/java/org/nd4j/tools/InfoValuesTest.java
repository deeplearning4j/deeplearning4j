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
