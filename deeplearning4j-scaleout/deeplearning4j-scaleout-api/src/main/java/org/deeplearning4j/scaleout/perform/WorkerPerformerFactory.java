package org.deeplearning4j.scaleout.perform;

import org.deeplearning4j.scaleout.conf.Configuration;

/**
 * Create a worker performer
 *
 * @author Adam Gibson
 */
public interface WorkerPerformerFactory {


    public final static String WORKER_PERFORMER = "org.deeplearning4j.scaleout.perform.workerperformer";

    /**
     * Create a worker performer
     * @return
     */
    public WorkerPerformer create();

    /**
     * Create based on the configuration
     * @param conf the configuration
     * @return the performer created based on the configuration
     */
    public WorkerPerformer create(Configuration conf);

}
