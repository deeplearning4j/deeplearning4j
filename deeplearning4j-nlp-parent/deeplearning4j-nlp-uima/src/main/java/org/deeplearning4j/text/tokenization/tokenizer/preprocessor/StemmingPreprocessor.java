package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

import org.tartarus.snowball.ext.PorterStemmer;

/**
 * This tokenizer preprocessor implements basic cleaning inherited from CommonPreprocessor + does english Porter stemming on tokens
 *
 * PLEASE NOTE: This preprocessor is thread-safe by using synchronized method
 *
 * @author raver119@gmail.com
 */
public class StemmingPreprocessor extends CommonPreprocessor {
    @Override
    public String preProcess(String token) {
        String prep = super.preProcess(token);
        PorterStemmer stemmer = new PorterStemmer();
        stemmer.setCurrent(prep);
        stemmer.stem();

        return stemmer.getCurrent();
    }
}
