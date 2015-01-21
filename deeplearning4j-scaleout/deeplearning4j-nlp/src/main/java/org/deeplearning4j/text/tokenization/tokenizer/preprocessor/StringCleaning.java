package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

/**
 * Various string cleaning utils
 * @author Adam GIbson
 */
public class StringCleaning {

    /**
     * Strip punctuation
     * @param base the base string
     * @return the cleaned string
     */
    public static String stripPunct(String base) {
        return base.replaceAll("[\\d\\.:,\"\'\\(\\)\\[\\]|/?!;]+","");

    }


}
