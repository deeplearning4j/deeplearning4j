package org.canova.cli.transforms.text.nlp;

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
