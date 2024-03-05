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

import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * An ArrayRegistry is a registry for {@link INDArray}
 * instances. This is mainly used for debugging and
 * profiling purposes.
 * <p>
 *     This registry is used for tracking arrays
 *     that are created and destroyed.
 *     <p>
 *         This registry is not persisted.
 *         <p>
 *             This registry is thread safe.
 *             <p>
 *
 */
public class DefaultArrayRegistry implements ArrayRegistry {

    private Map<Long, INDArray> arrays;
    private static AtomicBoolean callingFromContext = new AtomicBoolean(false);
    public DefaultArrayRegistry(Map<Long, INDArray> arrays) {
        this.arrays = arrays;
    }

    public DefaultArrayRegistry() {
        this.arrays = new ConcurrentHashMap<>();

    }




    @Override
    public Map<Long, INDArray> arrays() {
        return arrays;
    }

    @Override
    public INDArray lookup(long id) {
        return arrays.get(id);
    }

    @Override
    public void register(INDArray array) {
        if (callingFromContext.get())
            return;
        arrays.put(array.getId(), array);
    }

    @Override
    public boolean contains(long id) {
        return arrays.containsKey(id);
    }
}
