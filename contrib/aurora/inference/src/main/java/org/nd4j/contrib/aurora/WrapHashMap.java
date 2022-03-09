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

package org.nd4j.contrib.aurora;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.nd4j.linalg.api.ndarray.INDArray;

public class WrapHashMap<K extends INDArray, V> implements Map<INDArray, V> {

    private HashMap<WrapNDArray, V> map = new HashMap<>();

    @Override
    public void clear() {
        map.clear();

    }

    @Override
    public boolean containsKey(Object key) {
        return map.containsKey(new WrapNDArray((INDArray) key));
    }

    @Override
    public boolean containsValue(Object value) {
        return map.containsValue(value);
    }

    @Override
    public Set<Entry<INDArray, V>> entrySet() {
        throw new java.lang.UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public boolean isEmpty() {
        return map.isEmpty();
    }

    @Override
    public Set<INDArray> keySet() {
        Set<INDArray> ret = new HashSet<>();
        for (WrapNDArray x : map.keySet()) {
            ret.add(x.arr);
        }
        return ret;
    }

    @Override
    public V put(INDArray key, V value) {
        return map.put(new WrapNDArray((INDArray) key), value);
    }

    @Override
    public void putAll(Map<? extends INDArray, ? extends V> m) {
        for (Map.Entry<? extends INDArray, ? extends V> x : m.entrySet()) {
            this.put(x.getKey(), x.getValue());
        }

    }

    @Override
    public V remove(Object key) {
        return map.remove(new WrapNDArray((INDArray) key));
    }

    @Override
    public int size() {
        return map.size();
    }

    @Override
    public Collection<V> values() {
        return map.values();
    }

    @Override
    public V get(Object key) {
        return map.get(new WrapNDArray((INDArray) key));
    }

}
