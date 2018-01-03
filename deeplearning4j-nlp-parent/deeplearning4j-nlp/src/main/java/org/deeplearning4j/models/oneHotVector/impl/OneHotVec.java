package org.deeplearning4j.models.oneHotVector.impl;

import org.deeplearning4j.models.oneHotVector.api.WordVectorsAPI;
import org.deeplearning4j.models.oneHotVector.util.Histogram;
import org.deeplearning4j.models.oneHotVector.util.HistogramUtils;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareListSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class OneHotVec implements WordVectorsAPI {
    private static final Logger log = LoggerFactory.getLogger(OneHotVec.class);
    private static final int DEFAULTֹֹׁ_MIN_SUPPORT = 5;

    private static String UNK = "UNK";
    private Map<String, INDArray> dictionary = new HashMap<>();
    private int length = -1;


    public void createModel(LabelAwareSentenceIterator iter) throws IOException {
        createModel(iter, DEFAULTֹֹׁ_MIN_SUPPORT);
    }


    public OneHotVec createModel(LabelAwareSentenceIterator iter, int minSupport) throws IOException {
        TokenizerFactory tf = getTokenizer();
        new DefaultTokenizerFactory();
        tf.setTokenPreProcessor(new CommonPreprocessor());
        Histogram<String> terms = new Histogram<>();
        while (iter.hasNext()) {
            for (String token : tf.create(iter.nextSentence()).getTokens()) {
                terms.inc(token);
            }
        }
        HistogramUtils.minSupport(terms, minSupport);

        // put unknown word
        this.length = terms.size() + 1;
        dictionary.put(this.getUNK(), Nd4j.zeros(1, length).putScalar(0, 1));

        int i = 1;
        log.info(String.valueOf(terms.size()));
        for (String term : terms.keySet()) {
            INDArray vec = Nd4j.zeros(1, length);
            vec.putScalar(i++, 1);
            dictionary.put(term, vec);
        }
        return this;
    }

    /**
     * @param sentence
     * @return
     */
    public INDArray getSentenceVectorMatrix(String sentence) {
        TokenizerFactory tf = getTokenizer();
        List<String> tokens = tf.create(sentence).getTokens();
        if (tokens.isEmpty()) return null;

        String first = tokens.get(0);
        INDArray $ = dictionary.containsKey(first) ? dictionary.get(first) : dictionary.get(this.getUNK());
        for (int i = 1; i < tokens.size(); ++i) {
            String token = tokens.get(i);
            $.addi(dictionary.containsKey(first) ? dictionary.get(token) : dictionary.get(this.getUNK()));
        }
        return $;
    }

    @Override
    public String getUNK() {
        return this.UNK;
    }

    @Override
    public void setUNK(String newUNK) {
        this.UNK = newUNK;
    }

    @Override
    public boolean hasWord(String word) {
        return dictionary.containsKey(word);
    }

    @Override
    public INDArray getWordVectorMatrix(String word) {
        return dictionary.get(word);
    }

    @Override
    public INDArray getWordVectors(Collection<String> words) {
        TokenizerFactory tf = getTokenizer();
        List<String> tokens = new LinkedList<>();
        for (String word : words) {
            List<String> toks = tf.create(word).getTokens();
            if (!toks.isEmpty())
                tokens.add(toks.get(0));
        }
        if (tokens.isEmpty()) return null;

        String first = tokens.get(0);
        INDArray $ = dictionary.containsKey(first) ? dictionary.get(first) : dictionary.get(this.getUNK());
        for (int i = 1; i < tokens.size(); ++i) {
            String token = tokens.get(i);
            $.addi(dictionary.containsKey(token) ? dictionary.get(token) : dictionary.get(this.getUNK()));
        }
        return $;
    }

    public TokenizerFactory getTokenizer() {
        TokenizerFactory $ = new DefaultTokenizerFactory();
        $.setTokenPreProcessor(new CommonPreprocessor());
        return $;
    }

    /**
     * @return vector length, -1 if model was not initialized
     */
    public int getLength() {
        return this.length;
    }
}