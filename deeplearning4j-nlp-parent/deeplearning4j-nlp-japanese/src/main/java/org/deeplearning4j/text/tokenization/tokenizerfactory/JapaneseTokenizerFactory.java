package org.deeplearning4j.text.tokenization.tokenizerfactory;

import org.deeplearning4j.text.tokenization.tokenizer.JapaneseTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

import java.io.InputStream;

/**
 * Tokenizer Factory for Japanese based on Kuromoji.
 */
public class JapaneseTokenizerFactory implements TokenizerFactory {
    private final com.atilika.kuromoji.ipadic.Tokenizer kuromoji;
    private final boolean useBaseForm;
    private TokenPreProcess preProcessor;


    /**
     * Creates a factory that uses the defaults for Kuromoji.
     */
    public JapaneseTokenizerFactory() {
        this(new com.atilika.kuromoji.ipadic.Tokenizer.Builder(), false);
    }

    /**
     * Creates a factory that uses the Kuomoji defaults but returns normalized/stemmed "baseForm" tokens.
     * @param useBaseForm normalize conjugations "走った" -> "走る" instead of "走っ"
     */
    public JapaneseTokenizerFactory(boolean useBaseForm) {
        this(new com.atilika.kuromoji.ipadic.Tokenizer.Builder(), useBaseForm);
    }

    /**
     * Create a factory using the supplied Kuromoji configuration.  Use this for setting custom dictionaries or modes.
     * @param builder The Kuromoji Tokenizer builder with your configuration in place.
     */
    public JapaneseTokenizerFactory(com.atilika.kuromoji.ipadic.Tokenizer.Builder builder) {
        this(builder, false);
    }

    /**
     * Create a factory using the supplied Kuromoji configuration.  Use this for setting custom dictionaries or modes.
     * @param builder The Kuromoji Tokenizer builder with your configuration in place.
     * @param baseForm normalize conjugations "走った" -> "走る" instead of "走っ"
     */
    public JapaneseTokenizerFactory(com.atilika.kuromoji.ipadic.Tokenizer.Builder builder, boolean baseForm) {
        kuromoji = builder.build();
        useBaseForm = baseForm;
    }

    /**
     * Create a Tokenizer instance for the given sentence.
     *
     * Note: This method is thread-safe.
     * @param toTokenize the string to tokenize.
     * @return The tokenizer.
     */
    @Override
    public Tokenizer create(String toTokenize) {
        if (toTokenize.isEmpty()) {
            throw new IllegalArgumentException("Unable to proceed; no sentence to tokenize");
        }

        Tokenizer t = new JapaneseTokenizer(kuromoji, toTokenize, useBaseForm);

        if (preProcessor != null) {
            t.setTokenPreProcessor(preProcessor);
        }

        return t;
    }

    /**
     * InputStreams are currently unsupported.
     * @param toTokenize the string to tokenize.
     */
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

