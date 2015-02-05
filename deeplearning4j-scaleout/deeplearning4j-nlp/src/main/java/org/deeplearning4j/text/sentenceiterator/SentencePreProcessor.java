package org.deeplearning4j.text.sentenceiterator;

import java.io.Serializable;

/**
 * Sentence pre processor.
 * Used for pre processing strings
 *
 * @author Adam Gibson
 */
public interface SentencePreProcessor extends Serializable {

    /**
     * Pre process a sentence
     * @param sentence the sentence to pre process
     * @return the pre processed sentence
     */
	String preProcess(String sentence);
	
}
