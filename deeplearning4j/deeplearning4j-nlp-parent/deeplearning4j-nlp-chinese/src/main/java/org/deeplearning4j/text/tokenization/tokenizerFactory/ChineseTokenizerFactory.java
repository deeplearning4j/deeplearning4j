package org.deeplearning4j.text.tokenization.tokenizerFactory;

import org.deeplearning4j.text.tokenization.tokenizer.ChineseTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.InputStream;

/**
 * @date: June 2,2017
 * @author: wangfeng
 * @Description:
 */

public class ChineseTokenizerFactory implements TokenizerFactory {

    private TokenPreProcess tokenPreProcess;

    @Override
    public Tokenizer create(String toTokenize) {
        Tokenizer tokenizer = new ChineseTokenizer(toTokenize);
        tokenizer.setTokenPreProcessor(tokenPreProcess);
        return tokenizer;
    }

    @Override
    public Tokenizer create(InputStream toTokenize) {
        throw new UnsupportedOperationException();
        /*  Tokenizer t =  new ChineseStreamTokenizer(toTokenize);
        t.setTokenPreProcessor(tokenPreProcess);
        return t;*/
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess tokenPreProcess) {
        this.tokenPreProcess = tokenPreProcess;
    }

    @Override
    public TokenPreProcess getTokenPreProcessor() {
        return tokenPreProcess;
    }
}
