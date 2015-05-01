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

import java.io.Serializable;
import java.util.Collection;

/**
 *
 * A WorkerRetriever handles saving and loading
 * work for a given worker.
 *
 * This allows for scalable data access patterns
 * @author Adam Gibson
 */
public interface WorkRetriever extends Serializable {


    /**
     * Clears the worker
     * @param worker the worker to clear
     */
    void clear(String worker);

    /**
     * The collection of workers that are saved
     * @return the collection of workers that have data saved
     */
    Collection<String> workers();

    /**
     * Loads the data applyTransformToDestination
     * @param worker the worker to load for
     * @return the data for the given worker or null
     */
    Job load(String worker);

    /**
     * Saves the data applyTransformToDestination for a given worker
     * @param worker the worker to save data for
     * @param data the data to save
     */
    void save(String worker, Job data);


}
