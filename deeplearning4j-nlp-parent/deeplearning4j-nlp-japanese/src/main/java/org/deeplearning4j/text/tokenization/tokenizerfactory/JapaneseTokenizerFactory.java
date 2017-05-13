package org.deeplearning4j.text.tokenization.tokenizerfactory;

import org.deeplearning4j.text.tokenization.tokenizer.JapaneseTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

import java.io.InputStream;

public class JapaneseTokenizerFactory implements TokenizerFactory {
    private final com.atilika.kuromoji.ipadic.Tokenizer kuromoji;
    private final boolean useBaseForm;
    private TokenPreProcess preProcessor;

    public JapaneseTokenizerFactory() {
        this(new com.atilika.kuromoji.ipadic.Tokenizer.Builder(), false);
    }

    public JapaneseTokenizerFactory(boolean useBaseForm) {
        this(new com.atilika.kuromoji.ipadic.Tokenizer.Builder(), true);
    }

    public JapaneseTokenizerFactory(com.atilika.kuromoji.ipadic.Tokenizer.Builder builder) {
        this(builder, false);
    }

    public JapaneseTokenizerFactory(com.atilika.kuromoji.ipadic.Tokenizer.Builder builder, boolean baseForm) {
        kuromoji = builder.build();
        useBaseForm = baseForm;
    }

    @Override
    public Tokenizer create(String toTokenize) {
        if (toTokenize.isEmpty()) {
            throw new IllegalArgumentException("Unable to proceed; no sentence to tokenize");
        }

        JapaneseTokenizer t = new JapaneseTokenizer(kuromoji, toTokenize, useBaseForm);

        if (preProcessor != null) {
            t.setTokenPreProcessor(preProcessor);
        }

        return t;
    }

    @Override
    public Tokenizer create(InputStream toTokenize) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setTokenPreProcessor(TokenPreProcess preProcessor) {
        this.preProcessor = preProcessor;
    }

    @Override
    public TokenPreProcess getTokenPreProcessor() {
        return this.preProcessor;
    }

}

