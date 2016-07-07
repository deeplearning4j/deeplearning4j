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

public class NLPUtils {

	  /**
	   * Calc the IDF component
	   * 
	   */
	  public static double idf(double totalDocumentsInCorpus, double numberOfDocsTheTermAppearsIn) {
		  if ( totalDocumentsInCorpus > 0 ) {
			  return Math.log10( 1 + (totalDocumentsInCorpus / numberOfDocsTheTermAppearsIn) );
		  }
		  return 0;
	  }

	  /**
	   * Term frequency
	   * https://en.wikipedia.org/wiki/Tf%E2%80%93idf
	   *
	   */
	  public static double tf(int termFreqInSpecificDocuments) {
		  if (termFreqInSpecificDocuments > 0) {
			  return Math.log10(1 + termFreqInSpecificDocuments);
		  }
		  return 0;
	  }

	  /**
	   * Return td * idf
	   *
	   * @param tf  the term frequency (assumed calculated)
	   * @param idf inverse document frequency (assumed calculated)
	   * @return td * idf
	   */
	  public static double tfidf(double tf, double idf) {
	    return tf * idf;
	  }	
	
}
