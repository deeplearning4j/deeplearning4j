/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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
