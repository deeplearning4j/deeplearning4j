/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.scaleout.aggregator;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.scaleout.job.Job;

import java.io.Serializable;

/**
 *
 * Aggregate job results
 *
 * @author Adam Gibson
 */
public interface JobAggregator extends Serializable {


    String AGGREGATOR = "org.deeplearning4j.scaleout.aggregator";

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
