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

package org.nd4j.common.collections;

import lombok.*;

import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.*;

public class WeakIdentityHashMap<K, V> implements Map<K, V> {

    protected final Map<KeyRef<K>, V> map;
    protected final ReferenceQueue<K> refQueue;

    public WeakIdentityHashMap(){
        map = new HashMap<>();
        refQueue = new ReferenceQueue<>();
    }

    //Clear references to any map keys that have been GC'd
    protected void clearReferences(){
        Reference<? extends K> r;
        while((r = refQueue.poll()) != null){
            map.remove(r);
        }
    }

    @Override
    public int size() {
        clearReferences();
        return map.size();
    }

    @Override
    public boolean isEmpty() {
        clearReferences();
        return map.isEmpty();
    }

    @Override
    public boolean containsKey(Object key) {
        clearReferences();
        return map.containsKey(new KeyRef<>(key));
    }

    @Override
    public boolean containsValue(Object value) {
        clearReferences();
        return map.containsValue(value);
    }

    @Override
    public V get(Object key) {
        clearReferences();
        return map.get(new KeyRef<>(key));
    }

    @Override
    public V put(K key, V value) {
        clearReferences();
        map.put(new KeyRef<>(key), value);
        return value;
    }

    @Override
    public V remove(Object key) {
        clearReferences();
        return map.remove(new KeyRef<>(key));
    }

    @Override
    public void putAll(Map<? extends K, ? extends V> m) {
        clearReferences();
        for(Map.Entry<? extends K, ? extends V> e : m.entrySet()){
            map.put(new KeyRef<>(e.getKey()), e.getValue());
        }
    }

    @Override
    public void clear() {
        map.clear();
        clearReferences();
    }

    @Override
    public Set<K> keySet() {
        clearReferences();
        Set<K> ret = new HashSet<>();
        for(KeyRef<K> k : map.keySet() ){
            K key = k.get();
            if(key != null)
                ret.add(key);
        }
        return ret;
    }

    @Override
    public Collection<V> values() {
        clearReferences();
        return map.values();
    }

    @Override
    public Set<Map.Entry<K, V>> entrySet() {
        clearReferences();
        Set<Map.Entry<K, V>> ret = new HashSet<>();
        for(Map.Entry<KeyRef<K>, V> e : map.entrySet()){
            K k = e.getKey().get();
            if(k != null){
                ret.add(new Entry<K,V>(k, e.getValue()));
            }
        }
        return ret;
    }


    protected static class KeyRef<K> extends WeakReference<K> {
        private final int hash;
        public KeyRef(@NonNull K referent) {
            super(referent);
            this.hash = System.identityHashCode(referent);
        }

        @Override
        public int hashCode(){
            return hash;
        }

        @Override
        public boolean equals(Object o){
            if(this == o){
                return true;
            }
            if(o instanceof WeakReference){
                return this.get() == ((WeakReference) o).get();
            }
            return false;
        }
    }

    @Data
    @AllArgsConstructor
    protected static class Entry<K,V> implements Map.Entry<K, V> {
        protected K key;
        protected V value;

        @Override
        public V setValue(V value){
            this.value = value;
            return value;
        }
    }
}
