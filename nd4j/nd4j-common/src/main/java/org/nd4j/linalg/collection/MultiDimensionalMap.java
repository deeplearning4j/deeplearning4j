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

package org.nd4j.linalg.collection;

import org.nd4j.linalg.primitives.Pair;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListMap;

/**
 * Multiple key map
 */
public class MultiDimensionalMap<K, T, V> implements Serializable {

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
     * map contains more than <tt>Integer.MAX_VALUE</tt> elements, returns
     * <tt>Integer.MAX_VALUE</tt>.
     *
     * @return the number of key-value mappings in this map
     */
    public int size() {
        return backedMap.size();
    }

    /**
     * Returns <tt>true</tt> if this map contains no key-value mappings.
     *
     * @return <tt>true</tt> if this map contains no key-value mappings
     */
    public boolean isEmpty() {
        return backedMap.isEmpty();
    }

    /**
     * Returns <tt>true</tt> if this map contains a mapping for the specified
     * key.  More formally, returns <tt>true</tt> if and only if
     * this map contains a mapping for a key <tt>k</tt> such that
     * <tt>(key==null ? k==null : key.equals(k))</tt>.  (There can be
     * at most one such mapping.)
     *
     * @param key key whose presence in this map is to be tested
     * @return <tt>true</tt> if this map contains a mapping for the specified
     * key
     * @throws ClassCastException   if the key is of an inappropriate type for
     *                              this map
     *                              (<a href="Collection.html#optional-restrictions">optional</a>)
     * @throws NullPointerException if the specified key is null and this map
     *                              does not permit null keys
     *                              (<a href="Collection.html#optional-restrictions">optional</a>)
     */

    public boolean containsKey(Object key) {
        return backedMap.containsKey(key);
    }

    /**
     * Returns <tt>true</tt> if this map maps one or more keys to the
     * specified value.  More formally, returns <tt>true</tt> if and only if
     * this map contains at least one mapping to a value <tt>v</tt> such that
     * <tt>(value==null ? v==null : value.equals(v))</tt>.  This operation
     * will probably require time linear in the map size for most
     * implementations of the <tt>Map</tt> interface.
     *
     * @param value value whose presence in this map is to be tested
     * @return <tt>true</tt> if this map maps one or more keys to the
     * specified value
     * @throws ClassCastException   if the value is of an inappropriate type for
     *                              this map
     *                              (<a href="Collection.html#optional-restrictions">optional</a>)
     * @throws NullPointerException if the specified value is null and this
     *                              map does not permit null values
     *                              (<a href="Collection.html#optional-restrictions">optional</a>)
     */

    public boolean containsValue(Object value) {
        return backedMap.containsValue(value);
    }

    /**
     * Returns the value to which the specified key is mapped,
     * or {@code null} if this map contains no mapping for the key.
     * <p/>
     * <p>More formally, if this map contains a mapping from a key
     * {@code k} to a value {@code v} such that {@code (key==null ? k==null :
     * key.equals(k))}, then this method returns {@code v}; otherwise
     * it returns {@code null}.  (There can be at most one such mapping.)
     * <p/>
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
     *                              (<a href="Collection.html#optional-restrictions">optional</a>)
     * @throws NullPointerException if the specified key is null and this map
     *                              does not permit null keys
     *                              (<a href="Collection.html#optional-restrictions">optional</a>)
     */

    public V get(Object key) {
        return backedMap.get(key);
    }

    /**
     * Associates the specified value with the specified key in this map
     * (optional operation).  If the map previously contained a mapping for
     * the key, the old value is replaced by the specified value.  (A map
     * <tt>m</tt> is said to contain a mapping for a key <tt>k</tt> if and only
     * if {@link #containsKey(Object) m.containsKey(k)} would return
     * <tt>true</tt>.)
     *
     * @param key   key with which the specified value is to be associated
     * @param value value to be associated with the specified key
     * @return the previous value associated with <tt>key</tt>, or
     * <tt>null</tt> if there was no mapping for <tt>key</tt>.
     * (A <tt>null</tt> return can also indicate that the map
     * previously associated <tt>null</tt> with <tt>key</tt>,
     * if the implementation supports <tt>null</tt> values.)
     * @throws UnsupportedOperationException if the <tt>put</tt> operation
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
     * from key <tt>k</tt> to value <tt>v</tt> such that
     * <code>(key==null ?  k==null : key.equals(k))</code>, that mapping
     * is removed.  (The map can contain at most one such mapping.)
     * <p/>
     * <p>Returns the value to which this map previously associated the key,
     * or <tt>null</tt> if the map contained no mapping for the key.
     * <p/>
     * <p>If this map permits null values, then a return value of
     * <tt>null</tt> does not <i>necessarily</i> indicate that the map
     * contained no mapping for the key; it's also possible that the map
     * explicitly mapped the key to <tt>null</tt>.
     * <p/>
     * <p>The map will not contain a mapping for the specified key once the
     * call returns.
     *
     * @param key key whose mapping is to be removed from the map
     * @return the previous value associated with <tt>key</tt>, or
     * <tt>null</tt> if there was no mapping for <tt>key</tt>.
     * @throws UnsupportedOperationException if the <tt>remove</tt> operation
     *                                       is not supported by this map
     * @throws ClassCastException            if the key is of an inappropriate type for
     *                                       this map
     *                                       (<a href="Collection.html#optional-restrictions">optional</a>)
     * @throws NullPointerException          if the specified key is null and this
     *                                       map does not permit null keys
     *                                       (<a href="Collection.html#optional-restrictions">optional</a>)
     */

    public V remove(Object key) {
        return backedMap.remove(key);
    }

    /**
     * Copies all of the mappings from the specified map to this map
     * (optional operation).  The effect of this call is equivalent to that
     * of calling {@link Map<>#put(k, v)} on this map once
     * for each mapping from key <tt>k</tt> to value <tt>v</tt> in the
     * specified map.  The behavior of this operation is undefined if the
     * specified map is modified while the operation is in progress.
     *
     * @param m mappings to be stored in this map
     * @throws UnsupportedOperationException if the <tt>putAll</tt> operation
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
     * @throws UnsupportedOperationException if the <tt>clear</tt> operation
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
     * the iterator's own <tt>remove</tt> operation), the results of
     * the iteration are undefined.  The applyTransformToDestination supports element removal,
     * which removes the corresponding mapping from the map, via the
     * <tt>Iterator.remove</tt>, <tt>Set.remove</tt>,
     * <tt>removeAll</tt>, <tt>retainAll</tt>, and <tt>clear</tt>
     * operations.  It does not support the <tt>add</tt> or <tt>addAll</tt>
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
     * (except through the iterator's own <tt>remove</tt> operation),
     * the results of the iteration are undefined.  The collection
     * supports element removal, which removes the corresponding
     * mapping from the map, via the <tt>Iterator.remove</tt>,
     * <tt>Collection.remove</tt>, <tt>removeAll</tt>,
     * <tt>retainAll</tt> and <tt>clear</tt> operations.  It does not
     * support the <tt>add</tt> or <tt>addAll</tt> operations.
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
     * the iterator's own <tt>remove</tt> operation, or through the
     * <tt>setValue</tt> operation on a map entry returned by the
     * iterator) the results of the iteration are undefined.  The applyTransformToDestination
     * supports element removal, which removes the corresponding
     * mapping from the map, via the <tt>Iterator.remove</tt>,
     * <tt>Set.remove</tt>, <tt>removeAll</tt>, <tt>retainAll</tt> and
     * <tt>clear</tt> operations.  It does not support the
     * <tt>add</tt> or <tt>addAll</tt> operations.
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
         * removed from the map (by the iterator's <tt>remove</tt> operation).
         *
         * @param value new value to be stored in this entry
         * @return old value corresponding to the entry
         * @throws UnsupportedOperationException if the <tt>put</tt> operation
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
