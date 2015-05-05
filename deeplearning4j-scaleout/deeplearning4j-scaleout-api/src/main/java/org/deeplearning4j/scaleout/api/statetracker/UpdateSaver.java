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

/**
 * Saves an update and handles interfacing
 * with the state tracker for saving and loading
 * updates.
 */
public interface UpdateSaver extends Serializable {
    /**
     * Load the serialized element based on the passed in id
     * This will remove any reference to the specified id
     * and return the object
     * @param id the id of the object to retrieve
     * @return the laoded object
     * @throws Exception
     */
    Job load(String id) throws Exception;

    /**
     * Saves the passed in object for later loading
     * @param id the id of the object to save
     * @param save the object to save
     */
    void save(String id, Job save) throws Exception;

    /**
     * Cleans up the persistence layer.
     * This will usually be used to clear up left over files from updates
     */
    void cleanup();
}
