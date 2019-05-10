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

package org.deeplearning4j.models.glove.count;

import com.google.common.util.concurrent.AtomicDouble;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.linalg.primitives.Pair;

import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Drop-in replacement for CounterMap
 *
 * WORK IN PROGRESS, PLEASE DO NOT USE
 *
 * @author raver119@gmail.com
 */
public class CountMap<T extends SequenceElement> {
    private volatile Map<Pair<T, T>, AtomicDouble> backingMap = new ConcurrentHashMap<>();

    public CountMap() {
        // placeholder
    }

    public void incrementCount(T element1, T element2, double weight) {
        Pair<T, T> tempEntry = new Pair<>(element1, element2);
        if (backingMap.containsKey(tempEntry)) {
            backingMap.get(tempEntry).addAndGet(weight);
        } else {
            backingMap.put(tempEntry, new AtomicDouble(weight));
        }
    }

    public void removePair(T element1, T element2) {
        Pair<T, T> tempEntry = new Pair<>(element1, element2);
        backingMap.remove(tempEntry);
    }

    public void removePair(Pair<T, T> pair) {
        backingMap.remove(pair);
    }

    public double getCount(T element1, T element2) {
        Pair<T, T> tempEntry = new Pair<>(element1, element2);
        if (backingMap.containsKey(tempEntry)) {
            return backingMap.get(tempEntry).get();
        } else
            return 0;
    }

    public double getCount(Pair<T, T> pair) {
        if (backingMap.containsKey(pair)) {
            return backingMap.get(pair).get();
        } else
            return 0;
    }

    public Iterator<Pair<T, T>> getPairIterator() {
        return new Iterator<Pair<T, T>>() {
            private Iterator<Pair<T, T>> iterator = backingMap.keySet().iterator();

            @Override
            public boolean hasNext() {
                return iterator.hasNext();
            }

            @Override
            public Pair<T, T> next() {
                //MapEntry<T> entry = iterator.next();
                return iterator.next(); //new Pair<>(entry.getElement1(), entry.getElement2());
            }

            @Override
            public void remove() {
                throw new UnsupportedOperationException("remove() isn't supported here");
            }
        };
    }

    public int size() {
        return backingMap.size();
    }
}
