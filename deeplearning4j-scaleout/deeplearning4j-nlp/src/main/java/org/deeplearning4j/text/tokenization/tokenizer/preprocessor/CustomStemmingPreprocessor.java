package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

import lombok.NonNull;
import org.tartarus.snowball.SnowballProgram;

/**
 * This is StemmingPreprocessor compatible with different StemmingProcessors defined as lucene/tartarus SnowballProgram
 * Like, but not limited to: RussianStemmer, DutchStemmer, FrenchStemmer etc
 *
 * PLEASE NOTE: This preprocessor is NOT thread-safe.
 *
 * @author raver119@gmail.com
 */
public class CustomStemmingPreprocessor extends CommonPreprocessor {
    private SnowballProgram stemmer;
    public CustomStemmingPreprocessor(@NonNull SnowballProgram stemmer) {
        this.stemmer = stemmer;
    }

    @Override
    public String preProcess(String token) {
        String prep = super.preProcess(token);
        stemmer.setCurrent(prep);
        stemmer.stem();
        return stemmer.getCurrent();
    }
}
