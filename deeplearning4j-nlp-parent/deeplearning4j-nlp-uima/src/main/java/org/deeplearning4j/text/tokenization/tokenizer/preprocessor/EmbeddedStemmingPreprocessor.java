package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.tartarus.snowball.ext.PorterStemmer;

/**
 * This tokenizer preprocessor uses given preprocessor + does english Porter stemming on tokens on top of it
 *
 *
 * @author raver119@gmail.com
 */
@NoArgsConstructor
public class EmbeddedStemmingPreprocessor implements TokenPreProcess {
    private TokenPreProcess preProcessor;

    public EmbeddedStemmingPreprocessor(@NonNull TokenPreProcess preProcess) {
        this.preProcessor = preProcess;
    }

    @Override
    public String preProcess(String token) {
        String prep = preProcessor == null ? token : preProcessor.preProcess(token);
        PorterStemmer stemmer = new PorterStemmer();
        stemmer.setCurrent(prep);
        stemmer.stem();

        return stemmer.getCurrent();
    }
}
