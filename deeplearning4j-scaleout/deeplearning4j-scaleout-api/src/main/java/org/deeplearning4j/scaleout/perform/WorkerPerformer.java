package org.deeplearning4j.scaleout.perform;

import org.deeplearning4j.scaleout.conf.Configuration;
import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.job.Job;

/**
 * Performs a job
 *
 * @author Adam Gibson
 */
public interface WorkerPerformer extends DeepLearningConfigurable {


    /**
     * Perform a job
     * @param job the job to perform
     */
    void perform(Job job);


    /**
     * Update the job performer
     * @param o the objects to update with
     */
    void update(Object...o);

}
