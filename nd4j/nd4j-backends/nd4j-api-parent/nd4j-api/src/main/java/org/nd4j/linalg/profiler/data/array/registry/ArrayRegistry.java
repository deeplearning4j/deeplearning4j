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

import org.nd4j.linalg.api.ndarray.INDArray;

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
