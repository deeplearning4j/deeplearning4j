package org.deeplearning4j.models.rntn;

import org.apache.commons.math3.random.MersenneTwister;
import org.nd4j.linalg.api.activation.Activations;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.text.corpora.treeparser.TreeVectorizer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
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
    private TokenizerFactory tokenizerFactory;
    private String sentence = "<LABEL> This is one sentence. </LABEL>";
   @Before
   public void init() throws Exception {
       vectorizer = new TreeVectorizer();
       tokenizerFactory = new UimaTokenizerFactory(false);
       sentenceIter = new CollectionSentenceIterator(Arrays.asList(sentence));
       vec = new Word2Vec(sentenceIter);
       vec.fit();

   }


    @Test
    public void testRntn() throws Exception {


        RNTN rntn = new RNTN.Builder().setActivationFunction(Activations.tanh())
                .setAdagradResetFrequency(1).setCombineClassification(true).setFeatureVectors(vec)
                .setRandomFeatureVectors(false).setRng(new MersenneTwister(123))
                .setUseTensors(true).setNumHidden(25).build();
        List<Tree> trees = vectorizer.getTreesWithLabels(sentence,Arrays.asList("LABEL"));
        rntn.fit(trees);



    }


}
