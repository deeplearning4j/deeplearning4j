/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.common.tools;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;


import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.tools.SIS;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class SISTest {
	//
	//
	private SIS sis;
	//
	
	@Test
	public void testAll(@TempDir Path tmpFld) throws Exception {
		//
		sis = new SIS();
		//
		int mtLv = 0;
		//
		sis.initValues( mtLv, "TEST", System.out, System.err, tmpFld.getRoot().toAbsolutePath().toString(), "Test", "ABC", true, true );
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
	
	@AfterEach
	public void after() {
		//
		int mtLv = 0;
		if ( sis != null ) sis.onStop( mtLv );
		//
	//	tmpFld.delete();
		//
	}
	
	
	
}