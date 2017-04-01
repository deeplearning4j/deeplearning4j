/*-
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
 *
 */

package org.nd4j.linalg.api.instrumentation;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * Instrumentation for logging statistics
 * about the ndarrays being allocated and destroyed.
 *
 * @author Adam Gibson
 */
public interface Instrumentation {

    public final static String DESTROYED = "destroyed";
    public final static String CREATED = "created";

    /**
     * Log the given ndarray
     *
     * @param toLog  the ndarray to log
     * @param status the status
     */
    void log(INDArray toLog, String status);

    /**
     * Data buffer to log
     *
     * @param buffer the buffer to log
     * @param status the status
     */
    void log(DataBuffer buffer, String status);

    /**
     * Log the given ndarray
     *
     * @param toLog the ndarray to log
     */
    void log(INDArray toLog);

    /**
     * Data buffer to log
     *
     * @param buffer the buffer to log
     */
    void log(DataBuffer buffer);

    /**
     * Get the still alive ndarrays
     *
     * @return the still alive ndarrays
     */
    Collection<LogEntry> getStillAlive();

    /**
     * Get the destroyed ndarrays
     *
     * @return the destroyed ndarrays
     */
    Collection<LogEntry> getDestroyed();

    /**
     * Returns whether the given ndarray has been destroyed
     *
     * @param id the id to check
     * @return true if the ndarray has been destroyed, false otherwise
     */
    boolean isDestroyed(String id);


}
