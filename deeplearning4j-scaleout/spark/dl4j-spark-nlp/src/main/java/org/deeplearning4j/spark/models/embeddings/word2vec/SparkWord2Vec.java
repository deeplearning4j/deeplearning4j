package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectorsImpl;
import org.deeplearning4j.models.word2vec.*;

import java.io.Serializable;

/**
 * @author raver119@gmail.com
 */
public class SparkWord2Vec extends WordVectorsImpl<VocabWord> implements Serializable {

    protected SparkWord2Vec() {

    }

    public void train() {

    }

    public void train(JavaRDD<String> textCorpus) {

    }

    public static class Builder {
        protected int numEpochs;
        protected int iterations;


        public SparkWord2Vec build() {
            SparkWord2Vec w2v = new SparkWord2Vec();

            return w2v;
        }
    }
}
