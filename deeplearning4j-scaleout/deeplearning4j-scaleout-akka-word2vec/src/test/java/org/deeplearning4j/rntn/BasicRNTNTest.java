package org.deeplearning4j.rntn;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.text.treeparser.TreeVectorizer;
import org.deeplearning4j.word2vec.Word2Vec;
import org.deeplearning4j.word2vec.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.word2vec.sentenceiterator.SentenceIterator;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

/**
 * Created by agibsonccc on 7/3/14.
 */
public class BasicRNTNTest {

    private TreeVectorizer vectorizer;
    private Word2Vec vec;
    private SentenceIterator sentenceIter;
   @Before
   public void init() throws Exception {
       vectorizer = new TreeVectorizer();
       sentenceIter = new CollectionSentenceIterator(Arrays.asList("This is one sentence."));
       vec = new Word2Vec(sentenceIter);
       vec.train();

   }


    @Test
    public void testRntn() throws Exception {



        RNTN rntn = new RNTN.Builder().setActivationFunction(Activations.tanh())
                .setAdagradResetFrequency(1).setCombineClassification(true).setFeatureVectors(vec)
                .setRandomFeatureVectors(false).setRng(new MersenneTwister(123))
                .setUseTensors(true).setNumHidden(25).build();
        List<Tree> trees = vectorizer.getTrees("This is one sentence.");
        rntn.train(trees);



    }


}
