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

package org.deeplearning4j.scaleout.api.statetracker;


import org.deeplearning4j.scaleout.job.Job;

/**
 * Iterates and updates over the possible updates.
 * This is meant for use by the {@link org.deeplearning4j.iterativereduce.tracker.statetracker.UpdateSaver}
 * to handle iterating over the ids and doing something with the updates. This will usually be via the:
 * {@link org.deeplearning4j.iterativereduce.akka.DeepLearningAccumulator}
 * to handle collapsing updates avoiding having them all in memory at once
 * @author Adam Gibson
 */
public interface IterateAndUpdate  {


    /**
     * The accumulated result
     * @return the accumulated result
     */
    Job accumulated();
    /**
     * Accumulates the updates in to a result
     * by iterating over each possible worker
     * and obtaining the mini batch updates for each.
     */
    void accumulate() throws Exception;


}
