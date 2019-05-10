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

import java.util.*;

/**
 * Provides a wrapper for a {@link TreeSet}
 * that uses {@link IntArrayKeyMap.IntArray}
 * for proper comparison of int arrays
 * as keys.
 *
 * @author Adam Gibson
 */
public class IntArrayKeySet implements Set<int[]> {
    private Set<IntArrayKeyMap.IntArray> set = new LinkedHashSet<>();
    @Override
    public int size() {
        return set.size();
    }

    @Override
    public boolean isEmpty() {
        return set.isEmpty();
    }

    @Override
    public boolean contains(Object o) {
        return set.contains(new IntArrayKeyMap.IntArray((int[]) o));
    }

    @Override
    public Iterator<int[]> iterator() {
        List<int[]> ret = new ArrayList<>();
        for(IntArrayKeyMap.IntArray arr : set) {
            ret.add(arr.getBackingArray());
        }

        return ret.iterator();
    }

    @Override
    public Object[] toArray() {
        Object[] ret = new Object[size()];
        int count = 0;
        for(IntArrayKeyMap.IntArray intArray : set) {
            ret[count++] = intArray.getBackingArray();
        }

        return ret;
    }

    @Override
    public <T> T[] toArray(T[] ts) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean add(int[] ints) {
        return set.add(new IntArrayKeyMap.IntArray(ints));
    }

    @Override
    public boolean remove(Object o) {
        return set.remove(new IntArrayKeyMap.IntArray((int[]) o));
    }

    @Override
    public boolean containsAll(Collection<?> collection) {
        return set.containsAll(getCollection(collection));

    }

    @Override
    public boolean addAll(Collection<? extends int[]> collection) {
        return set.addAll(getCollection(collection));
    }

    @Override
    public boolean retainAll(Collection<?> collection) {
        return set.retainAll(getCollection(collection));
    }

    @Override
    public boolean removeAll(Collection<?> collection) {
        return set.removeAll(getCollection(collection));
    }

    @Override
    public void clear() {
        set.clear();
    }

    private Collection<IntArrayKeyMap.IntArray> getCollection(Collection<?> coll) {
        List<IntArrayKeyMap.IntArray> ret = new ArrayList<>();
        Collection<int[]> casted = (Collection<int[]>) coll;
        for(int[] arr : casted) {
            ret.add(new IntArrayKeyMap.IntArray(arr));
        }
        return ret;
    }

}
