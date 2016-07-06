package org.canova.cli.transforms.text.nlp;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import org.canova.api.conf.Configuration;
import org.canova.api.io.data.DoubleWritable;
import org.canova.api.io.data.Text;
import org.canova.api.records.reader.impl.FileRecordReader;
import org.canova.api.util.MathUtils;
import org.canova.api.writable.Writable;
import org.junit.Test;

public class TestTfidfTextVectorizerTransform {

	
	@Test
	public void testBasicOneSentenceTFIDF() {

		Configuration conf = new Configuration();
		conf.setInt( TfidfTextVectorizerTransform.MIN_WORD_FREQUENCY, 1 );

		TfidfTextVectorizerTransform tfidfTransform = new TfidfTextVectorizerTransform();
		tfidfTransform.initialize(conf);
		
		// 1. get some image data yo
		
		Collection<Writable> vector1 = new ArrayList<>();
		vector1.add(new Text("Go Dogs, Go") );
		vector1.add(new Text("label_A") );
	/*	
		Collection<Writable> vector2 = new ArrayList<>();
		vector2.add(new Text("go Dogs, Go Far") );
		vector2.add(new Text("label_B") );
*/
		
		tfidfTransform.collectStatistics(vector1);
	//	tfidfTransform.collectStatistics(vector2);
		
//		assertEquals( 0.0, normalizer.minValue, 0.0 );
//		assertEquals( 3.0, normalizer.maxValue, 0.0 );
		
		// 2. normalize it
		
		tfidfTransform.transform(vector1);
		
		// 3. check it
		
		assertEquals( 2, tfidfTransform.cache.vocabWords().size() );
//		assertEquals( 2, tfidfTransform.cache.vocabWords()
		System.out.println( tfidfTransform.cache.vocabWords().get(0).toString() );
		
		assertEquals( 3, vector1.size() );
		
		Iterator<Writable> iter = vector1.iterator();
		
		Double goCount = ((DoubleWritable)iter.next()).get();
		Double dogsCount = ((DoubleWritable)iter.next()).get();
		Double labelID = ((DoubleWritable)iter.next()).get();
		
//		assertEquals( 2.0, goCount, 0.0);
	//	assertEquals( 1.0, dogsCount, 0.0);
		assertEquals( 0.0, labelID, 0.0);
				
					
	
	
	}

}
