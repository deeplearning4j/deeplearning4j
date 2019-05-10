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

package org.nd4j.linalg.api.iter;

import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * Iterates and returns int arrays
 * over a particular shape.
 *
 * This iterator starts at zero and increments
 * the shape until each item in the "position"
 * hits the current shape
 *
 * @author Adam Gibson
 */
public class NdIndexIterator implements Iterator<long[]> {
    private int length = -1;
    private int i = 0;
    private long[] shape;
    private char order = 'c';
    private boolean cache = false;
    private static Map<Pair<long[], Character>, LinearIndexLookup> lookupMap = new HashMap<>();
    private LinearIndexLookup lookup;


    /**
     *  Pass in the shape to iterate over.
     *  Defaults to c ordering
     * @param shape the shape to iterate over
     */
    public NdIndexIterator(int... shape) {
        this('c', shape);
        this.cache = false;
    }

    public NdIndexIterator(long... shape) {
        this('c', false, shape);
        this.cache = false;
    }

    /**
     *  Pass in the shape to iterate over.
     *  Defaults to c ordering
     * @param shape the shape to iterate over
     */
    public NdIndexIterator(char order, boolean cache, long... shape) {
        this.shape = ArrayUtil.copy(shape);
        this.length = ArrayUtil.prod(shape);
        this.order = order;
        this.cache = cache;
        if (this.cache) {
            LinearIndexLookup lookup = lookupMap.get(new Pair<>(shape, order));
            if (lookup == null) {
                lookup = new LinearIndexLookup(shape, order);
                //warm up the cache
                for (int i = 0; i < length; i++) {
                    lookup.lookup(i);
                }
                lookupMap.put(new Pair<>(shape, order), lookup);
                this.lookup = lookup;
            } else {
                this.lookup = lookupMap.get(new Pair<>(shape, order));
            }

        }
    }

    /**
     *  Pass in the shape to iterate over
     * @param shape the shape to iterate over
     */
    public NdIndexIterator(char order, int... shape) {
        this(order, false, ArrayUtil.toLongArray(shape));
    }

    public NdIndexIterator(char order, long... shape) {
        this(order, false, shape);
    }

    @Override
    public boolean hasNext() {
        return i < length;
    }



    @Override
    public long[] next() {
        if (lookup != null)
            return lookup.lookup(i++);
        switch (order) {
            case 'c':
                return Shape.ind2subC(shape, i++);
            case 'f':
                return Shape.ind2sub(shape, i++);
            default:
                throw new IllegalArgumentException("Illegal ordering " + order);
        }

    }



    @Override
    public void remove() {

    }

}
