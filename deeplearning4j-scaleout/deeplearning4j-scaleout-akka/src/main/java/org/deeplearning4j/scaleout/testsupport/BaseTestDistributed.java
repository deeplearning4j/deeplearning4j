package org.deeplearning4j.scaleout.testsupport;

import org.deeplearning4j.scaleout.actor.runner.DeepLearning4jDistributed;
import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.job.JobIterator;
import org.deeplearning4j.scaleout.perform.WorkerPerformerFactory;
import org.deeplearning4j.scaleout.statetracker.StateTracker;
import org.deeplearning4j.scaleout.statetracker.hazelcast.HazelCastStateTracker;

/**
 * Baseline test support for testing a distributed app
 * @author Adam Gibson
 */
public abstract class BaseTestDistributed {
    protected JobIterator iterator;
    protected DeepLearning4jDistributed distributed;
    protected StateTracker stateTracker;
    protected String workPerformFactoryClassName;
    private Configuration conf;

    public void init() throws Exception {
        stateTracker = createStateTracker();
        iterator = createIterator();
        workPerformFactoryClassName = workPerformFactoryClassName();
        Configuration conf = createConfiguration();
        distributed = new DeepLearning4jDistributed(iterator,stateTracker);
        distributed.setup(conf);

    }


    public void tearDown() throws Exception {
        stateTracker.shutdown();
        distributed.shutdown();
    }


    public  Configuration createConfiguration() {
        Configuration conf = new Configuration();
        conf.set(WorkerPerformerFactory.WORKER_PERFORMER, workPerformFactoryClassName);
        return conf;
    }

    public StateTracker createStateTracker() throws  Exception {
        return new HazelCastStateTracker();
    }


    /**
     * Name of the work performer class
     * @return
     */
    public abstract String workPerformFactoryClassName();


    /**
     *
     * @return
     */
    public abstract JobIterator createIterator();


}
