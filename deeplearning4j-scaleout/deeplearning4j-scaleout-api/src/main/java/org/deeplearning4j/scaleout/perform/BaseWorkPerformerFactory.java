package org.deeplearning4j.scaleout.perform;

import org.deeplearning4j.nn.conf.Configuration;

/**
 * Base work performer factory
 * @author Adam Gibson
 */
public abstract class BaseWorkPerformerFactory implements  WorkerPerformerFactory {

    @Override
    public WorkerPerformer create() {
        Configuration conf = new Configuration();
        WorkerPerformer performer = instantiate();
        performer.setup(conf);
        return performer;
    }

    @Override
    public WorkerPerformer create(Configuration conf) {
        WorkerPerformer performer = instantiate();
        performer.setup(conf);
        return performer;
    }


    /**
     * Instantiate a work performer
     * @return
     */
    public abstract WorkerPerformer instantiate();

}
