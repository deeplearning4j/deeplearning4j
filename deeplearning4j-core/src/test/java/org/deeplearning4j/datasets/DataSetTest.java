/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.datasets;

import static org.junit.Assert.*;

import org.nd4j.linalg.dataset.DataSet;
import org.junit.Test;

public class DataSetTest {

	
	@Test
	public void testSetNewLabels() {
		DataSet d = DataSets.iris();
		DataSet filtered = d.filterBy(new int[]{2,3});
		d.filterAndStrip(new int[]{2,4});
		assertEquals(2,d.numOutcomes());
		assertEquals(filtered.numExamples(),d.numExamples());
		assertEquals(filtered.getFeatureMatrix(),d.getFeatureMatrix());
        assertEquals(filtered.numExamples(),d.getLabels().rows());
		
	
	}


	
	

}
