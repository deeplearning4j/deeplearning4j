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

import org.nd4j.autodiff.samediff.config.SDValue;

public class WrapHashMap<K extends SDValue, V> implements Map<SDValue, V> {

    private HashMap<WrapSDValue, V> map = new HashMap<>();

    @Override
    public void clear() {
        map.clear();

    }

    @Override
    public boolean containsKey(Object key) {
        return map.containsKey(new WrapSDValue((SDValue) key));
    }

    @Override
    public boolean containsValue(Object value) {
        return map.containsValue(value);
    }

    @Override
    public Set<Entry<SDValue, V>> entrySet() {
        throw new java.lang.UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public boolean isEmpty() {
        return map.isEmpty();
    }

    @Override
    public Set<SDValue> keySet() {
        Set<SDValue> ret = new HashSet<>();
        for (WrapSDValue x : map.keySet()) {
            ret.add(x.arr);
        }
        return ret;
    }

    @Override
    public V put(SDValue key, V value) {
        return map.put(new WrapSDValue((SDValue) key), value);
    }

    @Override
    public void putAll(Map<? extends SDValue, ? extends V> m) {
        for (Map.Entry<? extends SDValue, ? extends V> x : m.entrySet()) {
            this.put(x.getKey(), x.getValue());
        }

    }

    @Override
    public V remove(Object key) {
        return map.remove(new WrapSDValue((SDValue) key));
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
        return map.get(new WrapSDValue((SDValue) key));
    }

}
