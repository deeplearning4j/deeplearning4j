package org.deeplearning4j.text.documentiterator;

import java.io.InputStream;
import java.io.Serializable;


/**
 * Document Iterator: iterate over input streams
 * @author Adam Gibson
 *
 */
public interface DocumentIterator extends Serializable {



    /**
	 * Get the next document
	 * @return the input stream for the next document
	 */
	InputStream nextDocument();
	
	/**
	 * Whether there are anymore documents in the iterator
	 * @return whether there are anymore documents
	 * in the iterator
	 */
	boolean hasNext();
	
	/**
	 * Reset the iterator to the beginning
	 */
	void reset();
	
	
	

}
