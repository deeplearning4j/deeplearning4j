package org.deeplearning4j.spark.models.glove;

import org.apache.spark.api.java.function.VoidFunction;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.models.glove.GloveWeightLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;


/**
 * Base line word 2 vec performer
 *
 * @author Adam Gibson
 */
public class GlovePerformer implements VoidFunction<Triple<VocabWord,VocabWord,Double>> {


    public final static String NAME_SPACE = "org.deeplearning4j.scaleout.perform.models.glove";
    public final static String VECTOR_LENGTH = NAME_SPACE + ".length";
    public final static String ALPHA = NAME_SPACE + ".alpha";
    public final static String X_MAX = NAME_SPACE + ".xmax";
    public final static String MAX_COUNT = NAME_SPACE + ".maxcount";
    private GloveWeightLookupTable table;

    public GlovePerformer(GloveWeightLookupTable table) {
        this.table = table;
    }

    @Override
    public void call(Triple<VocabWord, VocabWord,Double> pair) throws Exception {
        table.iterateSample(pair.getFirst(),pair.getSecond(),pair.getThird());
    }
}
