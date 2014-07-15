package org.deeplearning4j.datasets;

import static org.junit.Assert.*;

import org.junit.Test;

public class DataSetTest {

	
	@Test
	public void testSetNewLabels() {
		DataSet d = DataSets.iris();
		DataSet filtered = d.filterBy(new int[]{2,3});
		d.filterAndStrip(new int[]{2,4});
		assertEquals(2,d.numOutcomes());
		assertEquals(filtered.numExamples(),d.numExamples());
		assertEquals(filtered.getFirst(),d.getFirst());
        assertEquals(filtered.numExamples(),d.getSecond().rows);
		
	
	}


	
	

}
