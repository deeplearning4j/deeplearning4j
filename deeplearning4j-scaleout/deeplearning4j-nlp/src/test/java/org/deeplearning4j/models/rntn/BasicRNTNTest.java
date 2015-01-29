package org.deeplearning4j.models.rntn;

import static org.junit.Assert.*;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.featuredetectors.autoencoder.recursive.Tree;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.invertedindex.LuceneInvertedIndex;
import org.junit.After;
import org.nd4j.linalg.api.activation.Activations;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.text.corpora.treeparser.TreeVectorizer;
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
    private InvertedIndex index;
    private VocabCache cache;
    private WeightLookupTable lookupTable;

    @Before
    public void init() throws Exception {
        new File("cache.ser").delete();
        FileUtils.deleteDirectory(new File("rntn-index"));

        if(vectorizer == null)
            vectorizer = new TreeVectorizer();
        if(tokenizerFactory == null)
            tokenizerFactory = new UimaTokenizerFactory(false);
        if(sentenceIter == null)
            sentenceIter = new CollectionSentenceIterator(Arrays.asList(sentence));
        File vectors = new File("wordvectors.ser");
        vectors.delete();
       if(cache == null)
            cache = new InMemoryLookupCache();
        if(lookupTable == null)
            lookupTable = new InMemoryLookupTable.Builder().cache(cache)
                    .vectorLength(100).build();

        if(index == null)
            index = new LuceneInvertedIndex.Builder()
                    .indexDir(new File("rntn-index")).cache(cache).build();
        if(vec == null) {
            vec = new Word2Vec.Builder()
                    .vocabCache(cache).index(index)
                    .iterate(sentenceIter).build();
            vec.fit();
        }



    }

    @After
    public void after() throws Exception {
        FileUtils.deleteDirectory(new File("rntn-index"));
    }


    @Test
    public void testGetValuesAndDerivativeLengths() throws Exception {

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

        List<Tree> trees = vectorizer.getTreesWithLabels(sentence,Arrays.asList("LABEL"));
        rntn.fit(trees);


    }



}
