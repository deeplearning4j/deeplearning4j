package org.deeplearning4j.scaleout.perform.models.word2vec;

import org.deeplearning4j.scaleout.perform.BaseWorkPerformerFactory;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;

/**
 * Work performer factory for word2vec
 * @author Adam Gibson
 */
public class Word2VecPerformerFactory extends BaseWorkPerformerFactory {


    @Override
    public WorkerPerformer instantiate() {
        return new Word2VecPerformer();
    }
}
