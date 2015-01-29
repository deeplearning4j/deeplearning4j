package org.deeplearning4j.scaleout.actor;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.scaleout.perform.WorkerPerformer;
import org.deeplearning4j.scaleout.perform.WorkerPerformerFactory;

/**
 * Created by agibsonccc on 11/27/14.
 */
public class TestPerformerFactory implements WorkerPerformerFactory {
    @Override
    public WorkerPerformer create() {
        return new TestPerformer();
    }

    @Override
    public WorkerPerformer create(Configuration conf) {
        return create();
    }
}
