package org.deeplearning4j.spark.text;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.util.Collection;

/**
 * Tokenizer function
 * @author Adam Gibson
 */
public class TokenizerFunction implements Function<String,Triple<Collection<String>,VocabCache,Long>> {
    private Class<? extends TokenizerFactory> tokenizerFactoryClazz;
    private TokenizerFactory tokenizerFactory;
    private VocabCache cache = new InMemoryLookupCache();

    public TokenizerFunction(String clazz,VocabCache cache) {
        try {
            tokenizerFactoryClazz = (Class<? extends TokenizerFactory>) Class.forName(clazz);
            this.cache = cache;
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

    }

    public TokenizerFunction() {
        this(DefaultTokenizerFactory.class.getName(),new InMemoryLookupCache());
    }
    public TokenizerFunction(VocabCache cache) {
        this(DefaultTokenizerFactory.class.getName(),cache);
    }

    @Override
    public Triple<Collection<String>,VocabCache,Long> call(String v1) throws Exception {
        Tokenizer tokenizer = null;
        if(tokenizerFactory == null)
            tokenizerFactory = getTokenizerFactory();
        Collection<String> tokens = tokenizerFactory.create(v1).getTokens();
        return new Triple<>(tokens,cache,Long.valueOf(tokens.size()));
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
