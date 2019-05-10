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

package org.nd4j.linalg.primitives;

import lombok.EqualsAndHashCode;

import java.io.Serializable;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 *
 * @author raver119@gmail.com
 */
@EqualsAndHashCode
public class CounterMap<F, S> implements Serializable{
    private static final long serialVersionUID = 119L;

    protected Map<F, Counter<S>> maps = new ConcurrentHashMap<>();

    public CounterMap() {

    }

    /**
     * This method checks if this CounterMap has any values stored
     *
     * @return
     */
    public boolean isEmpty() {
        return maps.isEmpty();
    }

    /**
     * This method checks if this CounterMap has any values stored for a given first element
     *
     * @param element
     * @return
     */
    public boolean isEmpty(F element){
        if (isEmpty())
            return true;

        Counter<S> m = maps.get(element);
        if (m == null)
            return true;
        else
            return m.isEmpty();
    }

    /**
     * This method will increment values of this counter, by counts of other counter
     *
     * @param other
     */
    public void incrementAll(CounterMap<F, S> other) {
        for (Map.Entry<F, Counter<S>> entry : other.maps.entrySet()) {
            F key = entry.getKey();
            Counter<S> innerCounter = entry.getValue();
            for (Map.Entry<S, AtomicDouble> innerEntry : innerCounter.entrySet()) {
                S value = innerEntry.getKey();
                incrementCount(key, value, innerEntry.getValue().get());
            }
        }
    }

    /**
     * This method will increment counts for a given first/second pair
     *
     * @param first
     * @param second
     * @param inc
     */
    public void incrementCount(F first, S second, double inc) {
        Counter<S> counter = maps.get(first);
        if (counter == null) {
            counter = new Counter<S>();
            maps.put(first, counter);
        }

        counter.incrementCount(second, inc);
    }

    /**
     * This method returns counts for a given first/second pair
     *
     * @param first
     * @param second
     * @return
     */
    public double getCount(F first, S second) {
        Counter<S> counter = maps.get(first);
        if (counter == null)
            return 0.0;

        return counter.getCount(second);
    }

    /**
     * This method allows you to set counter value for a given first/second pair
     *
     * @param first
     * @param second
     * @param value
     * @return
     */
    public double setCount(F first, S second, double value) {
        Counter<S> counter = maps.get(first);
        if (counter == null) {
            counter = new Counter<S>();
            maps.put(first, counter);
        }

        return counter.setCount(second, value);
    }

    /**
     * This method returns pair of elements with a max value
     *
     * @return
     */
    public Pair<F, S> argMax() {
        Double maxCount = -Double.MAX_VALUE;
        Pair<F, S> maxKey = null;
        for (Map.Entry<F, Counter<S>> entry : maps.entrySet()) {
            Counter<S> counter = entry.getValue();
            S localMax = counter.argMax();
            if (counter.getCount(localMax) > maxCount || maxKey == null) {
                maxKey = new Pair<F, S>(entry.getKey(), localMax);
                maxCount = counter.getCount(localMax);
            }
        }
        return maxKey;
    }

    /**
     * This method purges all counters
     */
    public void clear() {
        maps.clear();
    }

    /**
     * This method purges counter for a given first element
     * @param element
     */
    public void clear(F element) {
        Counter<S> s = maps.get(element);
        if (s != null)
            s.clear();
    }

    /**
     * This method returns Set of all first elements
     * @return
     */
    public Set<F> keySet() {
        return maps.keySet();
    }

    /**
     * This method returns counter for a given first element
     *
     * @param first
     * @return
     */
    public Counter<S> getCounter(F first) {
        return maps.get(first);
    }

    /**
     * This method returns Iterator of all first/second pairs stored in this counter
     *
     * @return
     */
    public Iterator<Pair<F, S>> getIterator() {
        return new Iterator<Pair<F, S>>() {

            Iterator<F> outerIt;
            Iterator<S> innerIt;
            F curKey;

            {
                outerIt = keySet().iterator();
            }

            private boolean hasInside() {
                if (innerIt == null || !innerIt.hasNext()) {
                    if (!outerIt.hasNext()) {
                        return false;
                    }
                    curKey = outerIt.next();
                    innerIt = getCounter(curKey).keySet().iterator();
                }
                return true;
            }

            public boolean hasNext() {
                return hasInside();
            }

            public Pair<F, S> next() {
                hasInside();
                if (curKey == null)
                    throw new RuntimeException("Outer element can't be null");

                return Pair.makePair(curKey, innerIt.next());
            }

            public void remove() {
                //
            }
        };
    }

    /**
     * This method returns number of First elements in this CounterMap
     * @return
     */
    public int size() {
        return maps.size();
    }

    /**
     * This method returns total number of elements in this CounterMap
     * @return
     */
    public int totalSize() {
        int size = 0;
        for (F first: keySet()) {
            size += getCounter(first).size();
        }

        return size;
    }
}
