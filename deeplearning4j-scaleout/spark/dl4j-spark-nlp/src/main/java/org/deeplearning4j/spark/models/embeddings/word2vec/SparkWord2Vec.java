package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.spark.models.embeddings.sequencevectors.SparkSequenceVectors;
import org.deeplearning4j.spark.models.embeddings.sequencevectors.functions.TokenizerFunction;

import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class SparkWord2Vec extends SparkSequenceVectors<VocabWord> {

    protected SparkWord2Vec() {

    }


    public void fitSentences(JavaRDD<String> sentences) {
        /**
         * Basically all we want here is tokenization, to get JavaRDD<Sequence<VocabWord>> out of Strings, and then we just go  for SeqVec
         */

        JavaRDD<Sequence<VocabWord>> seqRdd = sentences.map(new TokenizerFunction(configurationBroadcast));

        // now since we have new rdd - just pass it to SeqVec
        super.fitSequences(seqRdd);
    }
}
