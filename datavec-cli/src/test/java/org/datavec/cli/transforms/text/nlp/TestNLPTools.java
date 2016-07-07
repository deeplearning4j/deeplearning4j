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

import org.junit.Test;

public class TestNLPTools {

	@Test
	public void testBaseTFIDF() {
		
		// doc 1
		/*
		
		this	1
		is	1
		a	2
		sample	1
		
		*/
		
		// doc 2
		
		/*
		 
		this	1
		is	1
		another	2
		example	3
		  
		 */

		int totalDocs = 2; // N total documents in corpus

		
		// TF of "this"
		int termFreqInSpecificDoc_d1 = 1;
		
		// IDF
		int numberOfDocsTheTermAppearsIn_this = 2;

		double tf_this = NLPUtils.tf( termFreqInSpecificDoc_d1 );
		double idf_this = NLPUtils.idf( totalDocs, numberOfDocsTheTermAppearsIn_this );
		
		double tfidf_this = NLPUtils.tfidf(tf_this, idf_this);
		
		//assertEquals( 0.0, tfidf_this, 0.0 );
		
		System.out.println( "TD-IDF(this, d1): " + tfidf_this );

		// now test the word example in 
		
		// TF of "example" in doc2
		int termFreqInSpecificDoc_d2 = 3;
		
		// IDF
		int numberOfDocsTheTermAppearsIn_example = 1;

		double tf_example = NLPUtils.tf( termFreqInSpecificDoc_d2 );
		double idf_example = NLPUtils.idf( totalDocs, numberOfDocsTheTermAppearsIn_example );
		
		double tfidf_example = NLPUtils.tfidf(tf_example, idf_example);
		
		//assertEquals( 0.9030, tfidf_example, 0.01 );
		
		System.out.println( "TD-IDF(example, d2): " + tfidf_example );
		
		
	}

}
