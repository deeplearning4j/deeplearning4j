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

package org.deeplearning4j.scaleout.job;

/**
 * Job iterator
 *
 * @author Adam Gibson
 */
public interface JobIterator {


    /**
     * Assigns a worker id
     * @param workerId
     * @return
     */
    Job next(String workerId);

    /**
     * Get the next job
     * @return
     */
    Job next();


    /**
     * Whether there are anymore jobs
     * @return
     */
    boolean hasNext();

    /**
     * Reset to the beginning
     */
    void reset();

}
