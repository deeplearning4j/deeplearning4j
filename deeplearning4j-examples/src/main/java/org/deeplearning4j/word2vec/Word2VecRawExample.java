package org.deeplearning4j.word2vec;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.text.inputsanitation.InputHomogenization;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;

/**
 * Created by agibsonccc on 10/9/14.
 */
public class Word2VecRawExample {


    public static void main(String[] args) throws Exception {
        ClassPathResource resource = new ClassPathResource("raw_sentences.txt");
        SentenceIterator iter = new LineSentenceIterator(resource.getFile());
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

        //DocumentIterator iter = new FileDocumentIterator(resource.getFile());
        TokenizerFactory t = new DefaultTokenizerFactory();
        final EndingPreProcessor preProcessor = new EndingPreProcessor();
        t.setTokenPreProcessor(new TokenPreProcess() {
            @Override
            public String preProcess(String token) {
                token = token.toLowerCase();
                String base = preProcessor.preProcess(token);
                base = base.replaceAll("\\d","d");
                if(base.endsWith("ly") || base.endsWith("ing"))
                    System.out.println();
                return base;
            }
        });

        InMemoryLookupCache cache = new InMemoryLookupCache(100,false);
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1)
                .vocabCache(cache).iterations(5).learningRate(2.5e-2)
                .iterate(iter).tokenizerFactory(t).build();
        vec.fit();

        double sim = vec.similarity("day","night");
        Collection<String> similar = vec.wordsNearest("day",10);
        System.out.println(similar);

        //vec.plotTsne();
        Tsne tsne = new Tsne.Builder().setMaxIter(200)
                .learningRate(500).useAdaGrad(false)
                .normalize(false).usePca(false).build();
        cache.plotVocab(tsne);
        InMemoryLookupCache.writeTsneFormat(vec,tsne.getY(),new File("coords.csv"));

        for(VocabWord word : cache.vocabWords()) {
            System.out.println("Word " + word.getWord() + " codes " + Arrays.toString(word.getCodes()) + " points " + Arrays.toString(word.getPoints()) + " code length " + word.getCodeLength() + " word freq " + word.getWordFrequency());
        }




    }

}
