/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * Holds a set of arrays keyed by a String name, functioning essentially like a {@code Map<String,INDArray>}.<br>
 * Implementations may have different internal ways of storing arrays, however.<br>
 * For example for single threaded applications: {@link org.nd4j.autodiff.samediff.array.SingleThreadArrayHolder}<br>
 * And for multi-threaded: {@link org.nd4j.autodiff.samediff.array.ThreadSafeArrayHolder}
 *
 * @author Alex Black
 */
public interface ArrayHolder {

    /**
     * @return True if an array by that name exists
     */
    boolean hasArray(String name);

    /**
     * @param name Name of the array to get
     * @return The array, or null if no array with that name exists
     */
    INDArray getArray(String name);

    /**
     * Set the array for the specified name (new array, or replace if it already exists)
     *
     * @param name  Name of the array
     * @param array Array to set
     */
    void setArray(String name, INDArray array);

    /**
     * Remove the array from the ArrayHolder, returning it (if it exists)
     *
     * @param name Name of the array to return
     * @return The now-removed array
     */
    INDArray removeArray(String name);

    /**
     * @return Number of arrays in the ArrayHolder
     */
    int size();

    /**
     * Initialize from the specified array holder.
     * This clears all internal arrays, and adds all arrays from the specified array holder
     *
     * @param arrayHolder Array holder to initialize this based on
     */
    void initFrom(ArrayHolder arrayHolder);

    /**
     * @return Names of the arrays currently in the ArrayHolder
     */
    Collection<String> arrayNames();

    /**
     * Rename the entry with the specified name
     *
     * @param from Original name
     * @param to   New name
     */
    void rename(String from, String to);
}
