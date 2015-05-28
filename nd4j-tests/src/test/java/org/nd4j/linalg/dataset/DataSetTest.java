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

package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import static org.junit.Assert.assertEquals;

public class DataSetTest extends BaseNd4jTest {

	
	@Test
	public void testSetNewLabels() {
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;

		INDArray data = Nd4j.rand(10,10);
		INDArray labels = FeatureUtil.toOutcomeMatrix(new int[]{0, 0, 1, 1, 2, 2, 3, 3, 3, 3}, 4);
		DataSet d = new DataSet(data,labels);
		DataSet filtered = d.filterBy(new int[]{2,3});
		d.filterAndStrip(new int[]{2,3});
		assertEquals(2,d.numOutcomes());
		assertEquals(filtered.numExamples(),d.numExamples());
		assertEquals(filtered.getFeatureMatrix(),d.getFeatureMatrix());
        assertEquals(filtered.numExamples(),d.getLabels().rows());
		
	
	}


    @Override
    public char ordering() {
        return 'f';
    }
}
