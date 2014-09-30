package org.deeplearning4j.word2vec;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;

/**
 * Created by agibsonccc on 9/29/14.
 */
public class Word2VecPlotExample {
    private SentenceIterator iter;
    private TokenizerFactory tokenizer;
    private Word2Vec vec;

    public Word2VecPlotExample(String path) throws Exception {
        this.iter = UimaSentenceIterator.createWithPath(path);
        tokenizer = new DefaultTokenizerFactory();
    }



    public void train() throws Exception {
        VocabCache cache;
        if (vec == null && !new File("vec.ser").exists()) {
            cache = new InMemoryLookupCache(100, true, 0.025f);

            vec = new Word2Vec.Builder().minWordFrequency(5).vocabCache(cache)
                    .windowSize(5)
                    .layerSize(100).iterate(iter).tokenizerFactory(tokenizer)
                    .build();
            vec.setCache(cache);
            vec.fit();


            SerializationUtils.saveObject(vec, new File("vec.ser"));
            SerializationUtils.saveObject(cache, new File("cache.ser"));

        } else {
            vec = SerializationUtils.readObject(new File("vec.ser"));
            cache = SerializationUtils.readObject(new File("cache.ser"));
            vec.setCache(cache);
            for (String s : cache.words()) {
                System.out.println(s);
            }
            System.out.println("Plotting...");
            vec.plotTsne();
        }
    }




    public static void main(String[] args) throws Exception {
        new Word2VecPlotExample(args[0]).train();
    }


}
