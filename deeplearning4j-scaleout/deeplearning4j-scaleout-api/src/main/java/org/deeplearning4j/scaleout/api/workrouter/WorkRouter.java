package org.deeplearning4j.scaleout.api.workrouter;

import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.job.Job;

/**
 *
 * Handles routing of work
 *
 * @author Adam Gibson
 */
public interface WorkRouter extends DeepLearningConfigurable {

    public final static String NAME_SPACE = "org.deeplearning4j.scaleout.api.workrouter";
    public final static String WAIT_FOR_WORKERS = NAME_SPACE + ".wait";
    public final static String WORK_ROUTER = NAME_SPACE + ".workrouter";


    /**
     * Update the workers and master results
     */
    void update();

    /**
     * Whether to wait for workers or not
     * @return whether to wait for workers or not
     */
    boolean isWaitForWorkers();

    /**
     * Whether to send work or not
     * @return whether to send work or not
     */
    boolean sendWork();

    /**
     * The associated state tracker
     * for obtaining information about the cluster
     * @return the associated state tracker
     */
    StateTracker stateTracker();

    /**
     * Route a job using the state tracker
     * @param job the job to route
     */
    void routeJob(Job job);



}
