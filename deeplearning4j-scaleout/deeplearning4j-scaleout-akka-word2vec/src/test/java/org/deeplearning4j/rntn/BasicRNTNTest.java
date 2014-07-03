package org.deeplearning4j.rntn;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.text.treeparser.TreeVectorizer;
import org.junit.Before;
import org.junit.Test;

import java.util.List;

/**
 * Created by agibsonccc on 7/3/14.
 */
public class BasicRNTNTest {

    private TreeVectorizer vectorizer;

   @Before
   public void init() throws Exception {
       vectorizer = new TreeVectorizer();

   }


    @Test
    public void testRntn() throws Exception {
        RNTN rntn = new RNTN.Builder().setActivationFunction(Activations.tanh())
                .setAdagradResetFrequency(1).setCombineClassification(true)
                .setRandomFeatureVectors(false).setRng(new MersenneTwister(123))
                .setUseTensors(true).setNumHidden(25).build();
        List<Tree> trees = vectorizer.getTrees("I am Adam Gibson.");
        rntn.train(trees);



    }


}
