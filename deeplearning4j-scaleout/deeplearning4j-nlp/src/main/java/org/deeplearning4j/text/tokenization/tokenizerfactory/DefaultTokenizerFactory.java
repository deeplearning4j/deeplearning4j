package org.deeplearning4j.text.tokenization.tokenizerfactory;

import java.io.InputStream;

import org.deeplearning4j.text.tokenization.tokenizer.DefaultStreamTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.DefaultTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

/**
 * Default tokenizer based on string tokenizer or stream tokenizer
 * @author Adam Gibson
 */
public class DefaultTokenizerFactory implements TokenizerFactory {

    private TokenPreProcess tokenPreProcess;

    @Override
    public Tokenizer create(String toTokenize) {
        DefaultTokenizer t =  new DefaultTokenizer(toTokenize);
        t.setTokenPreProcessor(tokenPreProcess);
        return t;
    }

    @Override
    public Tokenizer create(InputStream toTokenize) {
        Tokenizer t =  new DefaultStreamTokenizer(toTokenize);
        t.setTokenPreProcessor(tokenPreProcess);
        return t;
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess preProcessor) {
        this.tokenPreProcess = preProcessor;
    }


}
