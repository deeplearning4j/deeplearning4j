package org.deeplearning4j.datasets;

import static org.junit.Assert.*;

import org.deeplearning4j.linalg.dataset.DataSet;
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
