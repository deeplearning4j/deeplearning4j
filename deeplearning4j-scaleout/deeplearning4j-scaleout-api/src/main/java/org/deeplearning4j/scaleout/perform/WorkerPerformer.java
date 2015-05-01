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

package org.deeplearning4j.scaleout.perform;

import org.deeplearning4j.nn.conf.DeepLearningConfigurable;
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
