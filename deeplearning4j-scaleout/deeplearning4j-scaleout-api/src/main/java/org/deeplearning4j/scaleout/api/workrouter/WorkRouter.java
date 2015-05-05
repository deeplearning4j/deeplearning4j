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

package org.deeplearning4j.scaleout.api.workrouter;

import org.deeplearning4j.scaleout.api.statetracker.StateTracker;
import org.deeplearning4j.nn.conf.DeepLearningConfigurable;
import org.deeplearning4j.scaleout.job.Job;

/**
 *
 * Handles routing of work
 *
 * @author Adam Gibson
 */
public interface WorkRouter extends DeepLearningConfigurable {

    String NAME_SPACE = "org.deeplearning4j.scaleout.api.workrouter";
    String WAIT_FOR_WORKERS = NAME_SPACE + ".wait";
    String WORK_ROUTER = NAME_SPACE + ".workrouter";


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
