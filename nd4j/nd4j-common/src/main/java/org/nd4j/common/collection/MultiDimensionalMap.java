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

package org.nd4j.common.collection;

import org.nd4j.common.primitives.Pair;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListMap;

/**
 * Multiple key map
 */
public class MultiDimensionalMap<K, T, V> implements Serializable {
    private static final long serialVersionUID = 1L;

    private Map<Pair<K, T>, V> backedMap;

    /**
     * Thread safe sorted map implementation
     * @param <K>
     * @param <T>
     * @param <V>
     * @return
     */
    public static <K, T, V> MultiDimensionalMap<K, T, V> newThreadSafeTreeBackedMap() {
        return new MultiDimensionalMap<>(new ConcurrentSkipListMap<Pair<K, T>, V>());
    }

    /**
     * Thread safe hash map implementation
     * @param <K>
     * @param <T>
     * @param <V>
     * @return
     */
    public static <K, T, V> MultiDimensionalMap<K, T, V> newThreadSafeHashBackedMap() {
        return new MultiDimensionalMap<>(new ConcurrentHashMap<Pair<K, T>, V>());
    }

    /**
     * Thread safe hash map impl
     * @param <K>
     * @param <T>
     * @param <V>
     * @return
     */
    public static <K, T, V> MultiDimensionalMap<K, T, V> newHashBackedMap() {
        return new MultiDimensionalMap<>(new HashMap<Pair<K, T>, V>());
    }

    /**
     * Tree map implementation
     * @param <K>
     * @param <T>
     * @param <V>
     * @return
     */
    public static <K, T, V> MultiDimensionalMap<K, T, V> newTreeBackedMap() {
        return new MultiDimensionalMap<>(new TreeMap<Pair<K, T>, V>());
    }

    public MultiDimensionalMap(Map<Pair<K, T>, V> backedMap) {
        this.backedMap = backedMap;
    }

    protected MultiDimensionalMap(){ }

    /**
     * Returns the number of key-value mappings in this map.  If the
     * map contains more than <code>Integer.MAX_VALUE</code> elements, returns
     * <code>Integer.MAX_VALUE</code>.
     *
     * @return the number of key-value mappings in this map
     */
    public int size() {
        return backedMap.size();
    }

    /**
     * Returns <code>true</code> if this map contains no key-value mappings.
     *
     * @return <code>true</code> if this map contains no key-value mappings
     */
    public boolean isEmpty() {
        return backedMap.isEmpty();
    }

    /**
     * Returns <code>true</code> if this map contains a mapping for the specified
     * key.  More formally, returns <code>true</code> if and only if
     * this map contains a mapping for a key <code>k</code> such that
     * <code>(key==null ? k==null : key.equals(k))</code>.  (There can be
     * at most one such mapping.)
     *
     * @param key key whose presence in this map is to be tested
     * @return <code>true</code> if this map contains a mapping for the specified
     * key
     * @throws ClassCastException   if the key is of an inappropriate type for
     *                              this map
     *                              ({@link Collection optional})
     * @throws NullPointerException if the specified key is null and this map
     *                              does not permit null keys
     *                              ({@link Collection optional})
     */

    public boolean containsKey(Object key) {
        return backedMap.containsKey(key);
    }

    /**
     * Returns <code>true</code> if this map maps one or more keys to the
     * specified value.  More formally, returns <code>true</code> if and only if
     * this map contains at least one mapping to a value <code>v</code> such that
     * <code>(value==null ? v==null : value.equals(v))</code>.  This operation
     * will probably require time linear in the map size for most
     * implementations of the <code>Map</code> interface.
     *
     * @param value value whose presence in this map is to be tested
     * @return <code>true</code> if this map maps one or more keys to the
     * specified value
     * @throws ClassCastException   if the value is of an inappropriate type for
     *                              this map
     *                              ({@link Collection optional})
     * @throws NullPointerException if the specified value is null and this
     *                              map does not permit null values
     *                              ({@link Collection optional})
     */

    public boolean containsValue(Object value) {
        return backedMap.containsValue(value);
    }

    /**
     * Returns the value to which the specified key is mapped,
     * or {@code null} if this map contains no mapping for the key.
     *
     * <p>More formally, if this map contains a mapping from a key
     * {@code k} to a value {@code v} such that {@code (key==null ? k==null :
     * key.equals(k))}, then this method returns {@code v}; otherwise
     * it returns {@code null}.  (There can be at most one such mapping.)
     *
     * <p>If this map permits null values, then a return value of
     * {@code null} does not <i>necessarily</i> indicate that the map
     * contains no mapping for the key; it's also possible that the map
     * explicitly maps the key to {@code null}.  The {@link #containsKey
     * containsKey} operation may be used to distinguish these two cases.
     *
     * @param key the key whose associated value is to be returned
     * @return the value to which the specified key is mapped, or
     * {@code null} if this map contains no mapping for the key
     * @throws ClassCastException   if the key is of an inappropriate type for
     *                              this map
     *                              ({@link Collection optional})
     * @throws NullPointerException if the specified key is null and this map
     *                              does not permit null keys
     *                              ({@link Collection optional})
     */

    public V get(Object key) {
        return backedMap.get(key);
    }

    /**
     * Associates the specified value with the specified key in this map
     * (optional operation).  If the map previously contained a mapping for
     * the key, the old value is replaced by the specified value.  (A map
     * <code>m</code> is said to contain a mapping for a key <code>k</code> if and only
     * if {@link #containsKey(Object) m.containsKey(k)} would return
     * <code>true</code>.)
     *
     * @param key   key with which the specified value is to be associated
     * @param value value to be associated with the specified key
     * @return the previous value associated with <code>key</code>, or
     * <code>null</code> if there was no mapping for <code>key</code>.
     * (A <code>null</code> return can also indicate that the map
     * previously associated <code>null</code> with <code>key</code>,
     * if the implementation supports <code>null</code> values.)
     * @throws UnsupportedOperationException if the <code>put</code> operation
     *                                       is not supported by this map
     * @throws ClassCastException            if the class of the specified key or value
     *                                       prevents it from being stored in this map
     * @throws NullPointerException          if the specified key or value is null
     *                                       and this map does not permit null keys or values
     * @throws IllegalArgumentException      if some property of the specified key
     *                                       or value prevents it from being stored in this map
     */

    public V put(Pair<K, T> key, V value) {
        return backedMap.put(key, value);
    }

    /**
     * Removes the mapping for a key from this map if it is present
     * (optional operation).   More formally, if this map contains a mapping
     * from key <code>k</code> to value <code>v</code> such that
     * <code>(key==null ?  k==null : key.equals(k))</code>, that mapping
     * is removed.  (The map can contain at most one such mapping.)
     *
     * <p>Returns the value to which this map previously associated the key,
     * or <code>null</code> if the map contained no mapping for the key.
     *
     * <p>If this map permits null values, then a return value of
     * <code>null</code> does not <i>necessarily</i> indicate that the map
     * contained no mapping for the key; it's also possible that the map
     * explicitly mapped the key to <code>null</code>.
     *
     * <p>The map will not contain a mapping for the specified key once the
     * call returns.
     *
     * @param key key whose mapping is to be removed from the map
     * @return the previous value associated with <code>key</code>, or
     * <code>null</code> if there was no mapping for <code>key</code>.
     * @throws UnsupportedOperationException if the <code>remove</code> operation
     *                                       is not supported by this map
     * @throws ClassCastException            if the key is of an inappropriate type for
     *                                       this map
     *                                       ({@link Collection optional})
     * @throws NullPointerException          if the specified key is null and this
     *                                       map does not permit null keys
     *                                       ({@link Collection optional})
     */

    public V remove(Object key) {
        return backedMap.remove(key);
    }

    /**
     * Copies all of the mappings from the specified map to this map
     * (optional operation).  The effect of this call is equivalent to that
     * of calling {@link Map#put(Object, Object) Map.put(k, v)} on this map once
     * for each mapping from key <code>k</code> to value <code>v</code> in the
     * specified map.  The behavior of this operation is undefined if the
     * specified map is modified while the operation is in progress.
     *
     * @param m mappings to be stored in this map
     * @throws UnsupportedOperationException if the <code>putAll</code> operation
     *                                       is not supported by this map
     * @throws ClassCastException            if the class of a key or value in the
     *                                       specified map prevents it from being stored in this map
     * @throws NullPointerException          if the specified map is null, or if
     *                                       this map does not permit null keys or values, and the
     *                                       specified map contains null keys or values
     * @throws IllegalArgumentException      if some property of a key or value in
     *                                       the specified map prevents it from being stored in this map
     */

    public void putAll(Map<? extends Pair<K, T>, ? extends V> m) {
        backedMap.putAll(m);
    }

    /**
     * Removes all of the mappings from this map (optional operation).
     * The map will be empty after this call returns.
     *
     * @throws UnsupportedOperationException if the <code>clear</code> operation
     *                                       is not supported by this map
     */

    public void clear() {
        backedMap.clear();
    }

    /**
     * Returns a {@link Set} view of the keys contained in this map.
     * The applyTransformToDestination is backed by the map, so changes to the map are
     * reflected in the applyTransformToDestination, and vice-versa.  If the map is modified
     * while an iteration over the applyTransformToDestination is in progress (except through
     * the iterator's own <code>remove</code> operation), the results of
     * the iteration are undefined.  The applyTransformToDestination supports element removal,
     * which removes the corresponding mapping from the map, via the
     * <code>Iterator.remove</code>, <code>Set.remove</code>,
     * <code>removeAll</code>, <code>retainAll</code>, and <code>clear</code>
     * operations.  It does not support the <code>add</code> or <code>addAll</code>
     * operations.
     *
     * @return a applyTransformToDestination view of the keys contained in this map
     */

    public Set<Pair<K, T>> keySet() {
        return backedMap.keySet();
    }

    /**
     * Returns a {@link Collection} view of the values contained in this map.
     * The collection is backed by the map, so changes to the map are
     * reflected in the collection, and vice-versa.  If the map is
     * modified while an iteration over the collection is in progress
     * (except through the iterator's own <code>remove</code> operation),
     * the results of the iteration are undefined.  The collection
     * supports element removal, which removes the corresponding
     * mapping from the map, via the <code>Iterator.remove</code>,
     * <code>Collection.remove</code>, <code>removeAll</code>,
     * <code>retainAll</code> and <code>clear</code> operations.  It does not
     * support the <code>add</code> or <code>addAll</code> operations.
     *
     * @return a collection view of the values contained in this map
     */

    public Collection<V> values() {
        return backedMap.values();
    }

    /**
     * Returns a {@link Set} view of the mappings contained in this map.
     * The applyTransformToDestination is backed by the map, so changes to the map are
     * reflected in the applyTransformToDestination, and vice-versa.  If the map is modified
     * while an iteration over the applyTransformToDestination is in progress (except through
     * the iterator's own <code>remove</code> operation, or through the
     * <code>setValue</code> operation on a map entry returned by the
     * iterator) the results of the iteration are undefined.  The applyTransformToDestination
     * supports element removal, which removes the corresponding
     * mapping from the map, via the <code>Iterator.remove</code>,
     * <code>Set.remove</code>, <code>removeAll</code>, <code>retainAll</code> and
     * <code>clear</code> operations.  It does not support the
     * <code>add</code> or <code>addAll</code> operations.
     *
     * @return a applyTransformToDestination view of the mappings contained in this map
     */

    public Set<Entry<K, T, V>> entrySet() {
        Set<Entry<K, T, V>> ret = new HashSet<>();
        for (Pair<K, T> pair : backedMap.keySet()) {
            ret.add(new Entry<>(pair.getFirst(), pair.getSecond(), backedMap.get(pair)));
        }
        return ret;
    }

    public V get(K k, T t) {
        return get(new Pair<>(k, t));
    }

    public void put(K k, T t, V v) {
        put(new Pair<>(k, t), v);
    }


    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (!(o instanceof MultiDimensionalMap))
            return false;

        MultiDimensionalMap that = (MultiDimensionalMap) o;

        return !(backedMap != null ? !backedMap.equals(that.backedMap) : that.backedMap != null);

    }


    public int hashCode() {
        return backedMap != null ? backedMap.hashCode() : 0;
    }


    public String toString() {
        return "MultiDimensionalMap{" + "backedMap=" + backedMap + '}';
    }


    public boolean contains(K k, T t) {
        return containsKey(new Pair<>(k, t));
    }


    public static class Entry<K, T, V> implements Map.Entry<Pair<K, T>, V> {

        private K firstKey;
        private T secondKey;
        private V value;

        public Entry(K firstKey, T secondKey, V value) {
            this.firstKey = firstKey;
            this.secondKey = secondKey;
            this.value = value;
        }

        public K getFirstKey() {
            return firstKey;
        }

        public void setFirstKey(K firstKey) {
            this.firstKey = firstKey;
        }

        public T getSecondKey() {
            return secondKey;
        }

        public void setSecondKey(T secondKey) {
            this.secondKey = secondKey;
        }

        public V getValue() {
            return value;
        }

        /**
         * Replaces the value corresponding to this entry with the specified
         * value (optional operation).  (Writes through to the map.)  The
         * behavior of this call is undefined if the mapping has already been
         * removed from the map (by the iterator's <code>remove</code> operation).
         *
         * @param value new value to be stored in this entry
         * @return old value corresponding to the entry
         * @throws UnsupportedOperationException if the <code>put</code> operation
         *                                       is not supported by the backing map
         * @throws ClassCastException            if the class of the specified value
         *                                       prevents it from being stored in the backing map
         * @throws NullPointerException          if the backing map does not permit
         *                                       null values, and the specified value is null
         * @throws IllegalArgumentException      if some property of this value
         *                                       prevents it from being stored in the backing map
         * @throws IllegalStateException         implementations may, but are not
         *                                       required to, throw this exception if the entry has been
         *                                       removed from the backing map.
         */

        public V setValue(V value) {
            V old = this.value;
            this.value = value;
            return old;
        }


        /**
         * Returns the key corresponding to this entry.
         *
         * @return the key corresponding to this entry
         * @throws IllegalStateException implementations may, but are not
         *                               required to, throw this exception if the entry has been
         *                               removed from the backing map.
         */

        public Pair<K, T> getKey() {
            return new Pair<>(firstKey, secondKey);
        }
    }



}
