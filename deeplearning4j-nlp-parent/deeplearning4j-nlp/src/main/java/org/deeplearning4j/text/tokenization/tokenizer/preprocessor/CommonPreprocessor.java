package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;

/**
 * @author jeffreytang
 */
public class CommonPreprocessor implements TokenPreProcess {
    @Override
    public String preProcess(String token) {
        return StringCleaning.stripPunct(token).toLowerCase();
    }
}
