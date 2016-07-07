/*
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.cli.transforms.text.nlp;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

import org.datavec.api.conf.Configuration;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
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
