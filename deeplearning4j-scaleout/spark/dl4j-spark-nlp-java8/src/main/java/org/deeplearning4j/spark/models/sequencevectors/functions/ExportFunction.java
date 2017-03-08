package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

/**
 * This function is used to
 *
 * @author raver119@gmail.com
 */
public class ExportFunction<T extends SequenceElement> implements VoidFunction<T> {

    public ExportFunction(Broadcast<VocabCache<ShallowSequenceElement>> vocabCacheBroadcast,
                    @NonNull String hdfsFilePath) {

    }

    @Override
    public void call(T t) throws Exception {

    }
}
