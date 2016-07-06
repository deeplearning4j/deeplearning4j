package org.canova.cli.transforms.image;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

import org.canova.api.io.data.DoubleWritable;
import org.canova.api.writable.Writable;
import org.junit.Test;

public class TestImageVectorNormalizeTransform {

	@Test
	public void testImageNormalizer() {
		
		NormalizeTransform normalizer = new NormalizeTransform();
		
		// 1. get some image data yo
		
		Collection<Writable> vector1 = new ArrayList<>();
		vector1.add(new DoubleWritable(1.0));
		vector1.add(new DoubleWritable(0.0));
		
		Collection<Writable> vector2 = new ArrayList<>();
		vector1.add(new DoubleWritable(3.0));
		vector1.add(new DoubleWritable(1.0));

		
		normalizer.collectStatistics(vector1);
		normalizer.collectStatistics(vector2);
		
		assertEquals( 0.0, normalizer.minValue, 0.0 );
		assertEquals( 3.0, normalizer.maxValue, 0.0 );
		
		// 2. normalize it
		
		normalizer.transform(vector1);
		
		// 3. check it
		
		Iterator<Writable> iter = vector1.iterator();
		
		Double val0 = ((DoubleWritable)iter.next()).get();
		Double val1 = ((DoubleWritable)iter.next()).get();
		
		assertEquals( (1.0/3.0), val0, 0.0);
		assertEquals(0.0, val1, 0.0);
		
	
	}

}
