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

package org.nd4j.linalg.util;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;

import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.common.tools.SIS;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;

@Slf4j
public class DataSetUtilsTest extends BaseNd4jTestWithBackends {


	@Override
	public char ordering(){
		return 'c';
	}

	//

	//
	private SIS sis;
	//
	@Test
	@ParameterizedTest
	@MethodSource("org.nd4j.linalg.BaseNd4jTest#configs")
	public void testAll(@TempDir Path tmpFld,Nd4jBackend backend) {
		//
		sis = new SIS();
		//
		int mtLv = 0;
		//
		sis.initValues( mtLv, "TEST", System.out, System.err, tmpFld.toAbsolutePath().toString(), "Test", "ABC", true, true );
		//
		INDArray in_INDA = Nd4j.zeros( 8, 8 );
		INDArray ot_INDA = Nd4j.ones( 8, 1 );
		//
		ot_INDA.putScalar( 7, 5 );
		//
		DataSet ds = new DataSet( in_INDA, ot_INDA );
		//
		DataSetUtils dl4jt = new DataSetUtils( sis, "TEST" );
		//
		dl4jt.showDataSet( mtLv, "ds", ds, 2, 2, 20, 20 );
		//
	//	assertEquals( 100, sis.getcharsCount() );
		//
		assertTrue( sis.getcharsCount() > 1190 && sis.getcharsCount() < 1210 );
		//
		INDArray spec_INDA = Nd4j.zeros( 8, 8 );
		//
		dl4jt.showINDArray( mtLv, "spec_INDA", spec_INDA, 3, 20, 20 );
		//
	//	assertEquals( 100, sis.getcharsCount() );
		//
		// this test might show different length on different systems due to various regional formatting options.
		assertTrue( sis.getcharsCount() > 2150 && sis.getcharsCount() < 2170 );
		//
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