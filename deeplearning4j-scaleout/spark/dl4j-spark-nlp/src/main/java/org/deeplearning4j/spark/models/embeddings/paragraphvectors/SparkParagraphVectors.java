package org.deeplearning4j.spark.models.embeddings.paragraphvectors;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.models.embeddings.paragraphvectors.functions.DocumentSequenceConvertFunction;
import org.deeplearning4j.spark.models.embeddings.paragraphvectors.functions.KeySequenceConvertFunction;
import org.deeplearning4j.spark.models.embeddings.sequencevectors.SparkSequenceVectors;
import org.deeplearning4j.text.documentiterator.LabelledDocument;

/**
 * @author raver119@gmail.com
 */
public class SparkParagraphVectors extends SparkSequenceVectors<VocabWord> {

    protected SparkParagraphVectors() {
        //
    }

    @Override
    protected VocabCache<ShallowSequenceElement> getShallowVocabCache() {
        return super.getShallowVocabCache();
    }


    /**
     * This method builds ParagraphVectors model, expecting JavaPairRDD with key as label, and value as document-in-a-string.
     *
     * @param documentsRdd
     */
    public void fitMultipleFiles(JavaPairRDD<String, String> documentsRdd) {
        /*
            All we want here, is to transform JavaPairRDD into JavaRDD<Sequence<VocabWord>>
         */
        broadcastEnvironment(new JavaSparkContext(documentsRdd.context()));

        JavaRDD<Sequence<VocabWord>> sequenceRdd = documentsRdd.map(new KeySequenceConvertFunction(configurationBroadcast));

        super.fitSequences(sequenceRdd);
    }

    /**
     * This method builds ParagraphVectors model, expecting JavaRDD<LabelledDocument>.
     * It can be either non-tokenized documents, or tokenized.
     *
     * @param documentsRdd
     */
    public void fitLabelledDocuments(JavaRDD<LabelledDocument> documentsRdd) {
        broadcastEnvironment(new JavaSparkContext(documentsRdd.context()));

        JavaRDD<Sequence<VocabWord>> sequenceRDD = documentsRdd.map(new DocumentSequenceConvertFunction(configurationBroadcast));

        super.fitSequences(sequenceRDD);
    }

}
