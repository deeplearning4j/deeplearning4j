/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.linalg.profiler.data.array.registry;

import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.profiler.data.stacktrace.StackTraceLineSkip;

import java.util.List;
import java.util.Map;


/**
 * An ArrayRegistry is a registry for {@link INDArray}
 * instances. This is mainly used for debugging and
 * profiling purposes.
 * <p>
 *     This registry is used for tracking arrays
 *     that are created and destroyed.
 *     <p>
 *         This registry is not persisted.
 */
public interface ArrayRegistry {


    /**
     * Returns a side by side comparison of each array
     * for the given session and index.
     *
     * @param session                   the session to compare
     * @param index                     the index of the array to compare
     * @param otherSession              the other session to compare
     * @param nextIndex                 the index of the array to compare for the other session
     * @param onlyCompareSameLineNumber whether to only compare arrays created the same line number
     * @param stackTraceLineSkipList
     * @return
     */
    Pair<List<NDArrayWithContext>,List<NDArrayWithContext>> compareArraysForSession(String session, int index, String otherSession, int nextIndex, boolean onlyCompareSameLineNumber, List<StackTraceLineSkip> stackTraceLineSkipList);

    /**
     * Returns the number of arrays registered
     * @param session the session to get the number of arrays for
     * @param index the index of the array to get
     * @return the number of arrays registered
     */
    String renderArraysForSession(String session, int index);

    /**
     * Returns the {@link INDArray}s registered
     * for a given session.
     * Each array is associated with a session
     * When an array is registered with a session
     * we can look up all arrays of the same session
     * and index to run comparisons.
     * <p>
     * An example is as follows:
     * enter test1
     * created array 1
     * index 0
     * exit test1
     * <p>
     * enter test1 (again)
     * created array 2
     * index 1
     * exit test1
     * <p>
     * Results: array 1  array 2
     *
     * @param session the session to get the arrays for
     * @param index   the index of the array to get
     * @return the {@link INDArray}s registered
     */
    List<NDArrayWithContext> arraysForSession(String session, int index);

    /**
     * This returns the  current count mentioned in
     * {@link #notifySessionEnter(String)}
     *  and {@link #notifySessionExit(String)}
     * @return
     */
    int numArraysRegisteredDuringSession();

    /**
     * When the {@link OpExecutioner#getExecutionTracker()}
     * is not null we track executions as part of a {@link org.nd4j.linalg.profiler.data.filter.OpExecutionEventSession}
     * which is created when we call {@link OpExecutionTracker#enterScope(String)}
     * which then calls {@link #notifySessionEnter(String)}.
     * When arrays are registered within a session we will track additional information about the array
     * by incrementing a counter and naming the array.
     * We use this to compare executions of the same arrays across different sessions.
     * @param sessionName
     */
    void notifySessionEnter(String sessionName);

    /**
     * When the {@link OpExecutioner#getExecutionTracker()}
     * is not null we track executions as part of a {@link org.nd4j.linalg.profiler.data.filter.OpExecutionEventSession}
     * which is created when we call {@link OpExecutionTracker#exitScope(String)} (String)}
     * which then calls {@link #notifySessionExit(String)} (String)}.
     * When arrays are registered within a session we will track additional information about the array
     * by incrementing a counter and naming the array.
     * When calling exit we reset this counter.
     * We use this to compare executions of the same arrays across different sessions.
     * @param sessionName
     */
    void notifySessionExit(String sessionName);
    
    /**
     * Returns all arrays registered
     * with this registry
     * @return
     */
    Map<Long,INDArray> arrays();

    /**
     * Returns the array with the given id
     * or null if it doesn't exist
     * @param id the id of the array to get
     * @return the array with the given id
     * or null if it doesn't exist
     */
    INDArray lookup(long id);


    /**
     * Returns true if the given array
     * is registered with this registry
     * @param array the array to check
     * @return true if the given array
     * is registered with this registry
     */

    void register(INDArray array);


    /**
     * Returns true if the given array
     * is registered with this registry
     *
     * @param array the array to check
     * @return true if the given array
     */
    default boolean contains(INDArray array) {
        return contains(array.getId());
    }

    /**
     * Returns true if the given array
     * is registered with this registry
     * @param id the id of the array to check
     * @return true if the given array
     */
    boolean contains(long id);
}
