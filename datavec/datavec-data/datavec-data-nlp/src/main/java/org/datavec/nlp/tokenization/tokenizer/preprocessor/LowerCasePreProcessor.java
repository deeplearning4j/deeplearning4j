package org.datavec.nlp.tokenization.tokenizer.preprocessor;

import org.datavec.nlp.tokenization.tokenizer.TokenPreProcess;

public class LowerCasePreProcessor implements TokenPreProcess {
    @Override
    public String preProcess(String token) {
        return token.toLowerCase();
    }
}
