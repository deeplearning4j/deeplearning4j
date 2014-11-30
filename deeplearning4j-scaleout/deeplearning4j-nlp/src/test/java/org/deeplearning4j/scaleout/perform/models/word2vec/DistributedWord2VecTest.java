package org.deeplearning4j.scaleout.perform.models.word2vec;

import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.scaleout.testsupport.BaseTestDistributed;

/**
 * Created by agibsonccc on 11/29/14.
 */
public class DistributedWord2VecTest extends BaseTestDistributed {
    @Override
    public String workPerformFactoryClassName() {
        return Word2VecPerformerFactory.class.getName();
    }

    @Override
    public JobIterator createIterator() {
        return null;
    }
}
