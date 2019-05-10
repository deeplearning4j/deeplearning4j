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

package org.nd4j.linalg.util;

import java.io.Serializable;
import java.util.*;

public class LinkedMultiValueMap<K, V> implements MultiValueMap<K, V>, Serializable {
    private static final long serialVersionUID = 3801124242820219131L;
    private final Map<K, List<V>> targetMap;

    public LinkedMultiValueMap() {
        this.targetMap = new LinkedHashMap();
    }

    public LinkedMultiValueMap(int initialCapacity) {
        this.targetMap = new LinkedHashMap(initialCapacity);
    }

    public LinkedMultiValueMap(Map<K, List<V>> otherMap) {
        this.targetMap = new LinkedHashMap(otherMap);
    }

    public void add(K key, V value) {
        List<V> values = this.targetMap.get(key);
        if (values == null) {
            values = new LinkedList<>();
            this.targetMap.put(key, values);
        }

        values.add(value);
    }

    public V getFirst(K key) {
        List<V> values = this.targetMap.get(key);
        return values != null ? values.get(0) : null;
    }

    public void set(K key, V value) {
        LinkedList values = new LinkedList();
        values.add(value);
        this.targetMap.put(key, values);
    }

    public void setAll(Map<K, V> values) {
        Iterator i$ = values.entrySet().iterator();

        while (i$.hasNext()) {
            Entry<K, V> entry = (Entry) i$.next();
            this.set(entry.getKey(), entry.getValue());
        }

    }

    public Map<K, V> toSingleValueMap() {
        LinkedHashMap singleValueMap = new LinkedHashMap(this.targetMap.size());
        Iterator i$ = this.targetMap.entrySet().iterator();

        while (i$.hasNext()) {
            Entry entry = (Entry) i$.next();
            singleValueMap.put(entry.getKey(), ((List) entry.getValue()).get(0));
        }

        return singleValueMap;
    }

    public int size() {
        return this.targetMap.size();
    }

    public boolean isEmpty() {
        return this.targetMap.isEmpty();
    }

    public boolean containsKey(Object key) {
        return this.targetMap.containsKey(key);
    }

    public boolean containsValue(Object value) {
        return this.targetMap.containsValue(value);
    }

    public List<V> get(Object key) {
        return this.targetMap.get(key);
    }

    public List<V> put(K key, List<V> value) {
        return this.targetMap.put(key, value);
    }

    public List<V> remove(Object key) {
        return this.targetMap.remove(key);
    }

    public void putAll(Map<? extends K, ? extends List<V>> m) {
        this.targetMap.putAll(m);
    }

    public void clear() {
        this.targetMap.clear();
    }

    public Set<K> keySet() {
        return this.targetMap.keySet();
    }

    public Collection<List<V>> values() {
        return this.targetMap.values();
    }

    public Set<Entry<K, List<V>>> entrySet() {
        return this.targetMap.entrySet();
    }

    public boolean equals(Object obj) {
        return this.targetMap.equals(obj);
    }

    public int hashCode() {
        return this.targetMap.hashCode();
    }

    public String toString() {
        return this.targetMap.toString();
    }
}
