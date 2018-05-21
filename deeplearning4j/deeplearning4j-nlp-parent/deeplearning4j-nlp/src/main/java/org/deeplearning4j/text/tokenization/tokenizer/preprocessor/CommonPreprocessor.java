package org.deeplearning4j.text.tokenization.tokenizer.preprocessor;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;

/**
 * A ToeknPreProcess implementation that removes puncuation marks and lower-cases.
 * <br>
 * Note that the implementation uses String#toLowerCase(String) and its behavior depends on the default locale.
 * @see StringCleaning#stripPunct(String)
 * @author jeffreytang
 */
public class CommonPreprocessor implements TokenPreProcess {
    @Override
    public String preProcess(String token) {
        return StringCleaning.stripPunct(token).toLowerCase();
    }
}
