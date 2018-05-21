package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

import lombok.NonNull;
import org.tartarus.snowball.SnowballProgram;

/**
 * This is StemmingPreprocessor compatible with different StemmingProcessors defined as lucene/tartarus SnowballProgram
 * such as: RussianStemmer, DutchStemmer, FrenchStemmer etc.
 * <br>
 * Note that CommonPreprocessor#preProcess(String) is first applied (i.e. punctuation marks are removed and and lower-cased), then the stemmer is applied.
 * <br>
 * This preprocessor is synchronized, thus thread-safe.
 *
 * @author raver119@gmail.com
 */
public class CustomStemmingPreprocessor extends CommonPreprocessor {
    private SnowballProgram stemmer;

    public CustomStemmingPreprocessor(@NonNull SnowballProgram stemmer) {
        this.stemmer = stemmer;
    }

    @Override
    public synchronized String preProcess(String token) {
        String prep = super.preProcess(token);
        stemmer.setCurrent(prep);
        stemmer.stem();
        return stemmer.getCurrent();
    }
}
