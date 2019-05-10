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

package org.nd4j.autodiff.execution.input;

import lombok.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;

/**
 *  This class
 * @author raver119@gmail.com
 */
public class Operands {
    private Map<NodeDescriptor, INDArray> map = new LinkedHashMap<>();

    /**
     * This method allows to pass array to the node identified by its name
     *
     * @param id
     * @param array
     * @return
     */
    public Operands addArgument(@NonNull String id, @NonNull INDArray array) {
        map.put(NodeDescriptor.builder().name(id).build(), array);
        return this;
    }

    /**
     * This method allows to pass array to the node identified by numeric id
     *
     * @param id
     * @param array
     * @return
     */
    public Operands addArgument(int id, @NonNull INDArray array) {
        map.put(NodeDescriptor.builder().id(id).build(), array);
        return this;
    }

    /**
     * This method allows to pass array to multi-output node in the graph
     *
     * @param id
     * @param index
     * @param array
     * @return
     */
    public Operands addArgument( int id, int index, @NonNull INDArray array) {
        map.put(NodeDescriptor.builder().id(id).index(index).build(), array);
        return this;
    }

    /**
     * This method allows to pass array to multi-output node in the graph
     *
     * @param id
     * @param index
     * @param array
     * @return
     */
    public Operands addArgument(String name, int id, int index, @NonNull INDArray array) {
        map.put(NodeDescriptor.builder().name(name).id(id).index(index).build(), array);
        return this;
    }

    /**
     * This method returns array identified its name
     * @param name
     * @return
     */
    public INDArray getById(@NonNull String name) {
        return map.get(NodeDescriptor.builder().name(name).build());
    }

    /**
     * This method returns array identified its numeric id
     * @param name
     * @return
     */
    public INDArray getById(int id) {
        return map.get(NodeDescriptor.builder().id(id).build());
    }

    /**
     * This method returns array identified its numeric id and index
     * @param name
     * @return
     */
    public INDArray getById(int id, int index) {
        return map.get(NodeDescriptor.builder().id(id).index(index).build());
    }

    /**
     * This method return operands as array, in order of addition
     * @return
     */
    public INDArray[] asArray() {
        val val = map.values();
        val res = new INDArray[val.size()];
        int cnt = 0;
        for (val v: val)
            res[cnt++] = v;

        return res;
    }

    /**
     * This method returns contents of this entity as collection of key->value pairs
     * @return
     */
    public Collection<Pair<NodeDescriptor, INDArray>> asCollection() {
        val c = new HashSet<Pair<NodeDescriptor, INDArray>>();
        for (val k: map.keySet())
            c.add(Pair.makePair(k, map.get(k)));

        return c;
    }

    /**
     * This method returns number of values in this entity
     * @return
     */
    public int size() {
        return map.size();
    }

    @NoArgsConstructor
    @AllArgsConstructor
    @Builder
    @Data
    public static class NodeDescriptor {
        private String name;
        private int id;
        private int index;
    }
}
