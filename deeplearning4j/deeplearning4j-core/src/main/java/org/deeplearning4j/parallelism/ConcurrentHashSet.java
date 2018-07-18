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

package org.deeplearning4j.parallelism;

import lombok.NonNull;

import java.io.Serializable;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * This is simplified ConcurrentHashSet implementation
 *
 * PLEASE NOTE: This class does NOT implement real equals & hashCode
 * 
 * @deprecated Please use {@code Collections.newSetFromMap(new ConcurrentHashMap<>())}
 *
 * @author raver119@gmail.com
 */
@Deprecated
public class ConcurrentHashSet<E> implements Set<E>, Serializable {
    private static final long serialVersionUID = 123456789L;

    // we're using concurrenthashmap behind the scenes
    private ConcurrentHashMap<E, Boolean> backingMap;



    public ConcurrentHashSet() {
        backingMap = new ConcurrentHashMap<>();
    }

    public ConcurrentHashSet(@NonNull Collection<E> collection) {
        this();
        addAll(collection);
    }


    @Override
    public int size() {
        return backingMap.size();
    }

    @Override
    public boolean isEmpty() {
        return backingMap.isEmpty();
    }

    @Override
    public boolean contains(Object o) {
        return backingMap.containsKey(o);
    }

    @Override
    public Iterator<E> iterator() {
        return new Iterator<E>() {
            private Iterator<Map.Entry<E, Boolean>> iterator = backingMap.entrySet().iterator();

            @Override
            public boolean hasNext() {
                return iterator.hasNext();
            }

            @Override
            public E next() {
                return iterator.next().getKey();
            }

            @Override
            public void remove() {
                // do nothing
            }
        };
    }

    @Override
    public Object[] toArray() {
        throw new UnsupportedOperationException();
    }

    @Override
    public <T> T[] toArray(T[] a) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean add(@NonNull E e) {
        Boolean ret = backingMap.putIfAbsent(e, Boolean.TRUE);

        return ret == null;
    }

    @Override
    public boolean remove(Object o) {
        return backingMap.remove(o);
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        for (Object o : c) {
            if (!contains(o))
                return false;
        }
        return true;
    }

    @Override
    public boolean addAll(Collection<? extends E> c) {
        for (E e : c)
            add(e);

        return true;
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        return false;
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        for (Object o : c)
            remove(o);

        return true;
    }

    @Override
    public void clear() {
        backingMap.clear();
    }
}
