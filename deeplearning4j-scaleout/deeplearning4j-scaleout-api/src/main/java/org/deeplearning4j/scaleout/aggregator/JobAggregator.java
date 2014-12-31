package org.deeplearning4j.scaleout.aggregator;

import org.deeplearning4j.nn.conf.Configuration;
import org.deeplearning4j.scaleout.job.Job;

import java.io.Serializable;

/**
 *
 * Aggregate job results
 *
 * @author Adam Gibson
 */
public interface JobAggregator extends Serializable {


    public final static String AGGREGATOR = "org.deeplearning4j.scaleout.aggregator";

    /**
     * Accumulate results of a job
     * @param job
     */
    void accumulate(Job job);

    /**
     * Return the aggregate results of a job
     * @return
     */
    Job aggregate();


    /**
     * Initialize based on the configuration
     * @param conf
     */
    void init(Configuration conf);



}
