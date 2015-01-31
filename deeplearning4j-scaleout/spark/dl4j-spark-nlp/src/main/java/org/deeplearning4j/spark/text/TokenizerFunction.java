package org.deeplearning4j.spark.text;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.Collection;
import java.util.List;

/**
 * Tokenizer function
 * @author Adam Gibson
 */
public class TokenizerFunction implements Function<String,Pair<List<String>,Long>> {
    private Class<? extends TokenizerFactory> tokenizerFactoryClazz;
    private TokenizerFactory tokenizerFactory;

    public TokenizerFunction(String clazz) {
        try {
            tokenizerFactoryClazz = (Class<? extends TokenizerFactory>) Class.forName(clazz);
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

    }

    public TokenizerFunction() {
        this(DefaultTokenizerFactory.class.getName());
    }

    @Override
    public Pair<List<String>,Long> call(String v1) throws Exception {
        if(tokenizerFactory == null)
            tokenizerFactory = getTokenizerFactory();
        List<String> tokens = tokenizerFactory.create(v1).getTokens();
        return new Pair<>(tokens,Long.valueOf(tokens.size()));
    }
    private TokenizerFactory getTokenizerFactory() {
        try {
            tokenizerFactory = tokenizerFactoryClazz.newInstance();
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }
        return tokenizerFactory;
    }

}
