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

package org.nd4j.contrib.aurora;

import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import org.nd4j.autodiff.samediff.config.SDValue;

public class WrapHashSet<K extends SDValue> implements Set<SDValue> {

    public HashSet<WrapSDValue> set = new HashSet<>();

    @Override
    public boolean add(SDValue e) {
        return set.add(new WrapSDValue(e));
    }

    @Override
    public boolean addAll(Collection<? extends SDValue> c) {
        c.forEach(x -> add(x));
        return false;
    }


    @Override
    public void clear() {
        set.clear();

    }

    @Override
    public boolean contains(Object o) {
        return set.contains(new WrapSDValue((SDValue) o));
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        for (Object h : c) {
            if (!contains(h))
                return false;
        }
        return true;
    }

    @Override
    public boolean isEmpty() {
        return set.isEmpty();
    }

    @Override
    public Iterator<SDValue> iterator() {
        return new InnerIterator(set.iterator());
    }

    @Override
    public boolean remove(Object o) {
        return set.remove(new WrapSDValue((SDValue) o));
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        for (Object h : c) {
            if (!remove(h))
                return false;
        }
        return true;
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        throw new java.lang.UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public int size() {
        return set.size();
    }

    @Override
    public Object[] toArray() {
        int s = 0;
        SDValue[] to = new SDValue[size()];
        for (WrapSDValue x : set) {
            to[s] = x.arr;
            ++s;
        }
        return null;
    }

    @Override
    public <T> T[] toArray(T[] a) {
        throw new java.lang.UnsupportedOperationException("Not supported yet.");
    }

    public static class InnerIterator implements Iterator<SDValue> {

        private Iterator<WrapSDValue> it;

        public InnerIterator(Iterator<WrapSDValue> it) {
            this.it = it;
        }

        @Override
        public boolean hasNext() {
            return it.hasNext();
        }

        @Override
        public SDValue next() {
            WrapSDValue x = it.next();
            return x.arr;
        }

    }

}
