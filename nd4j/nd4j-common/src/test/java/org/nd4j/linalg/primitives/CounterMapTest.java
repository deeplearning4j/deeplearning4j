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

import org.junit.Test;

import java.util.Iterator;

import static org.junit.Assert.*;

/**
 * CounterMap tests
 *
 * @author raver119@gmail.com
 */
public class CounterMapTest {

    @Test
    public void testIterator() {
        CounterMap<Integer, Integer> counterMap = new CounterMap<>();

        counterMap.incrementCount(0, 0, 1);
        counterMap.incrementCount(0, 1, 1);
        counterMap.incrementCount(0, 2, 1);
        counterMap.incrementCount(1, 0, 1);
        counterMap.incrementCount(1, 1, 1);
        counterMap.incrementCount(1, 2, 1);

        Iterator<Pair<Integer, Integer>> iterator = counterMap.getIterator();

        Pair<Integer, Integer> pair = iterator.next();

        assertEquals(0, pair.getFirst().intValue());
        assertEquals(0, pair.getSecond().intValue());

        pair = iterator.next();

        assertEquals(0, pair.getFirst().intValue());
        assertEquals(1, pair.getSecond().intValue());

        pair = iterator.next();

        assertEquals(0, pair.getFirst().intValue());
        assertEquals(2, pair.getSecond().intValue());

        pair = iterator.next();

        assertEquals(1, pair.getFirst().intValue());
        assertEquals(0, pair.getSecond().intValue());

        pair = iterator.next();

        assertEquals(1, pair.getFirst().intValue());
        assertEquals(1, pair.getSecond().intValue());

        pair = iterator.next();

        assertEquals(1, pair.getFirst().intValue());
        assertEquals(2, pair.getSecond().intValue());


        assertFalse(iterator.hasNext());
    }


    @Test
    public void testIncrementAll() {
        CounterMap<Integer, Integer> counterMapA = new CounterMap<>();

        counterMapA.incrementCount(0, 0, 1);
        counterMapA.incrementCount(0, 1, 1);
        counterMapA.incrementCount(0, 2, 1);
        counterMapA.incrementCount(1, 0, 1);
        counterMapA.incrementCount(1, 1, 1);
        counterMapA.incrementCount(1, 2, 1);

        CounterMap<Integer, Integer> counterMapB = new CounterMap<>();

        counterMapB.incrementCount(1, 1, 1);
        counterMapB.incrementCount(2, 1, 1);

        counterMapA.incrementAll(counterMapB);

        assertEquals(2.0, counterMapA.getCount(1,1), 1e-5);
        assertEquals(1.0, counterMapA.getCount(2,1), 1e-5);
        assertEquals(1.0, counterMapA.getCount(0,0), 1e-5);


        assertEquals(7, counterMapA.totalSize());


        counterMapA.setCount(2, 1, 17);

        assertEquals(17.0, counterMapA.getCount(2, 1), 1e-5);
    }
}