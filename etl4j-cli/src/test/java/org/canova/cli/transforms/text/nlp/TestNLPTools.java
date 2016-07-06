package org.canova.cli.transforms.text.nlp;

import static org.junit.Assert.*;

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
