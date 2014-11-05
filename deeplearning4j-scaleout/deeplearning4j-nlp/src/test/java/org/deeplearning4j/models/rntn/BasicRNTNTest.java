package org.deeplearning4j.models.rntn;

import static org.junit.Assert.*;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.activation.Activations;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.text.corpora.treeparser.TreeVectorizer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.Arrays;
import java.util.List;

/**
 * Created by agibsonccc on 7/3/14.
 */
public class BasicRNTNTest {

    private TreeVectorizer vectorizer;
    private Word2Vec vec;
    private SentenceIterator sentenceIter;
    private TokenizerFactory tokenizerFactory;
    private String sentence = "<LABEL> This is one sentence. </LABEL>";
    @Before
    public void init() throws Exception {
        new File("cache.ser").delete();
        vectorizer = new TreeVectorizer();
        tokenizerFactory = new UimaTokenizerFactory(false);
        sentenceIter = new CollectionSentenceIterator(Arrays.asList(sentence));
        File vectors = new File("wordvectors.ser");
        vectors.delete();
        vec = new Word2Vec.Builder().vocabCache(new InMemoryLookupCache.Builder().vectorLength(100).build())
                .iterate(sentenceIter).build();
        vec.fit();




    }

    @Test
    public void testGetValuesAndDerivativeLengths() {
        RNTN rntn = new RNTN.Builder().setActivationFunction(Activations.tanh())
                .setAdagradResetFrequency(1)
                .setCombineClassification(true).setFeatureVectors(vec)
                .setRandomFeatureVectors(false)
                .setRng(new MersenneTwister(123))
                .setUseTensors(true).build();
        INDArray params = rntn.getParameters();
        INDArray gradient = rntn.getValueGradient();
        rntn.setParameters(params);
        assertEquals(params.length(),gradient.length());
    }

    @Test
    public void testRntn() throws Exception {


        RNTN rntn = new RNTN.Builder().setActivationFunction(Activations.tanh())
                .setAdagradResetFrequency(1)
                .setCombineClassification(true).setFeatureVectors(vec)
                .setRandomFeatureVectors(false)
                .setRng(new MersenneTwister(123))
                .setUseTensors(true).build();
        List<Tree> trees = vectorizer.getTreesWithLabels(sentence,Arrays.asList("LABEL"));
        rntn.fit(trees);



    }


}
