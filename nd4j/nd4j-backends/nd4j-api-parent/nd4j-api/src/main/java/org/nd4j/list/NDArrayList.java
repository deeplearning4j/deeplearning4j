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

package org.nd4j.list;

import lombok.NonNull;
import lombok.val;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.EqualsCondition;

import java.util.*;

/**
 * An {@link ArrayList} like implementation of {@link List}
 * using {@link INDArray} as the backing data structure
 *
 * @author Adam Gibson
 */
public class NDArrayList extends  BaseNDArrayList<Double>  {
    private INDArray container;
    private int size;

    public NDArrayList() {
        this.container = Nd4j.create(10L);
    }

    /**
     * Specify the underlying ndarray for this list.
     * @param container the underlying array.
     */
    public NDArrayList(@NonNull INDArray container) {
        Preconditions.checkState(container == null || container.rank() == 1, "Container must be rank 1: is rank %s",
                container == null ? 0 : container.rank());
        this.container = container;
    }


    /**
     * Get a view of the underlying array
     * relative to the size of the actual array.
     * (Sometimes there are overflows in the internals
     * but you want to use the internal INDArray for computing something
     * directly, this gives you the relevant subset that reflects the content of the list)
     * @return the view of the underlying ndarray relative to the collection's real size
     */
    public INDArray array() {
        if(isEmpty()) {
            throw new ND4JIllegalStateException("Array is empty!");
        }

         return container.get(NDArrayIndex.interval(0,size));
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public boolean isEmpty() {
        return size == 0;
    }

    @Override
    public boolean contains(Object o) {
        return indexOf(o) >= 0;
    }

    @Override
    public Iterator<Double> iterator() {
        return new NDArrayListIterator();
    }

    @Override
    public Object[] toArray() {
        throw new UnsupportedOperationException();
    }

    @Override
    public <T> T[] toArray(T[] ts) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean add(Double aDouble) {
        if(container == null) {
            container = Nd4j.create(10L);
        }
        else if(size == container.length()) {
            INDArray newContainer = Nd4j.create(container.length() * 2L);
            newContainer.put(new INDArrayIndex[]{NDArrayIndex.interval(0,container.length())},container);
            container = newContainer;
        }

        container.putScalar(size++,aDouble);
        return true;
    }

    @Override
    public boolean remove(Object o) {
        int idx = BooleanIndexing.firstIndex(container,new EqualsCondition((double) o)).getInt(0);
        if(idx < 0)
            return false;
        container.put(new INDArrayIndex[]{NDArrayIndex.interval(idx,container.length())},container.get(NDArrayIndex.interval(idx + 1,container.length())));
        container = container.reshape(size);
        return true;
    }

    @Override
    public boolean containsAll(Collection<?> collection) {
        for(Object d : collection) {
            if(!contains(d)) {
                return false;
            }
        }

        return true;
    }

    @Override
    public boolean addAll(Collection<? extends Double> collection) {
        if(collection instanceof NDArrayList) {
            NDArrayList ndArrayList = (NDArrayList) collection;
            ndArrayList.growCapacity(this.size() + collection.size());

        }
        else {
            for(Double d : collection) {
                add(d);
            }
        }
        return true;
    }

    @Override
    public boolean addAll(int i, Collection<? extends Double> collection) {

        for(Double d : collection) {
            add(i,d);
        }

        return true;
    }

    @Override
    public boolean removeAll(Collection<?> collection) {
        for(Object d : collection) {
            remove(d);
        }

        return true;
    }

    @Override
    public boolean retainAll(Collection<?> collection) {
        return false;
    }

    @Override
    public void clear() {
        size = 0;
        container = null;
    }

    @Override
    public Double get(int i) {
        return container.getDouble(i);
    }

    @Override
    public Double set(int i, Double aDouble) {
        container.putScalar(i,aDouble);
        return aDouble;
    }

    @Override
    public void add(int i, Double aDouble) {
        rangeCheck(i);
        growCapacity(i);
        moveForward(i);
        container.putScalar(i,aDouble);
        size++;

    }

    @Override
    public Double remove(int i) {
        rangeCheck(i);
        int numMoved = this.size - i - 1;
        if(numMoved > 0) {
            double move = container.getDouble(i);
            moveBackward(i);
            size--;
            return move;
        }

        return null;
    }

    @Override
    public int indexOf(Object o) {
        return BooleanIndexing.firstIndex(container,new EqualsCondition((double) o)).getInt(0);
    }

    @Override
    public int lastIndexOf(Object o) {
        return BooleanIndexing.lastIndex(container,new EqualsCondition((double) o)).getInt(0);
    }

    @Override
    public ListIterator<Double> listIterator() {
        return new NDArrayListIterator();
    }

    @Override
    public ListIterator<Double> listIterator(int i) {
        return new NDArrayListIterator(i);
    }

    @Override
    public List<Double> subList(int i, int i1) {
        return new NDArrayList(container.get(NDArrayIndex.interval(i,i1)));
    }

    @Override
    public String toString() {
        return container.get(NDArrayIndex.interval(0,size)).toString();
    }

    private class NDArrayListIterator implements ListIterator<Double> {
        private int curr = 0;

        public NDArrayListIterator(int curr) {
            this.curr = curr;
        }

        public NDArrayListIterator() {
        }

        @Override
        public boolean hasNext() {
            return curr < size;
        }

        @Override
        public Double next() {
            double ret = get(curr);
            curr++;
            return ret;
        }

        @Override
        public boolean hasPrevious() {
            return curr > 0;
        }

        @Override
        public Double previous() {
            double ret = get(curr - 1);
            curr--;
            return ret;
        }

        @Override
        public int nextIndex() {
            return curr + 1;
        }

        @Override
        public int previousIndex() {
            return curr - 1;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException();

        }

        @Override
        public void set(Double aDouble) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void add(Double aDouble) {
            throw new UnsupportedOperationException();
        }
    }



    private void growCapacity(int idx) {
        if(container == null) {
            container = Nd4j.create(10L);
        }
        else if(idx >= container.length()) {
            long max = Math.max(container.length() * 2L,idx);
            INDArray newContainer = Nd4j.create(max);
            newContainer.put(new INDArrayIndex[]{NDArrayIndex.interval(0,container.length())},container);
            container = newContainer;
        }
    }



    private void rangeCheck(int idx) {
        if(idx < 0 || idx > size) {
            throw new IllegalArgumentException("Illegal index " + idx);
        }
    }

    private void moveBackward(int index) {
        int numMoved = size - index - 1;
        INDArrayIndex[] first = new INDArrayIndex[] {NDArrayIndex.interval(index  ,index  + numMoved)};
        INDArrayIndex[] getRange = new INDArrayIndex[] {NDArrayIndex.interval(index + 1 ,index + 1  + numMoved)};
        container.put(first,container.get(getRange));
    }

    private void moveForward(int index) {
        int numMoved = size - index - 1;
        INDArrayIndex[] getRange = new INDArrayIndex[] {NDArrayIndex.interval(index,index + numMoved)};
        INDArray get = container.get(getRange).dup();
        INDArrayIndex[] first = new INDArrayIndex[] {NDArrayIndex.interval(index + 1,index + 1 + get.length())};
        container.put(first,get);
    }

}
