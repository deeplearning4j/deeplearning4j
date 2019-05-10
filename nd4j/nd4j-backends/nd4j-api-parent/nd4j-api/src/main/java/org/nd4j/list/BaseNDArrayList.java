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

import lombok.val;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
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
public abstract  class BaseNDArrayList<X extends Number> extends  AbstractList<X>  {
    protected INDArray container;
    protected int size;



    public BaseNDArrayList() {
        this.container = Nd4j.create(10);
    }

    /**
     * Specify the underlying ndarray for this list.
     * @param container the underlying array.
     */
    public BaseNDArrayList(INDArray container) {
        this.container = container;
    }

    /**
     * Allocates the container and this list with
     * the given size
     * @param size the size to allocate with
     */
    public void allocateWithSize(int size) {
        container = Nd4j.create(1,size);
        this.size = size;
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
        return container.get(NDArrayIndex.interval(0,size)).reshape(1,size);
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
    public Iterator<X> iterator() {
        return new NDArrayListIterator();
    }

    @Override
    public Object[] toArray() {
        Number number = get(0);
        if(number instanceof Integer) {
            Integer[] ret = new Integer[size()];
            for(int i = 0; i < ret.length; i++) {
                ret[i] = (Integer) get(i);
            }

            return ret;
        }
        else if(number instanceof Double) {
            Double[] ret = new Double[size()];
            for(int i = 0; i < ret.length; i++) {
                ret[i] = (Double) get(i);
            }

            return ret;
        }
        else if(number instanceof Float) {
            Float[] ret = new Float[size()];
            for(int i = 0; i < ret.length; i++) {
                ret[i] = (Float) get(i);
            }

            return ret;
        }

        throw new UnsupportedOperationException("Unable to convert to array");
    }

    @Override
    public <T> T[] toArray(T[] ts) {
        Number number = get(0);
        if(number instanceof Integer) {
            Integer[] ret = (Integer[]) ts;
            for(int i = 0; i < ret.length; i++) {
                ret[i] = (Integer) get(i);
            }

            return (T[]) ret;
        }
        else if(number instanceof Double) {
            Double[] ret = new Double[size()];
            for(int i = 0; i < ret.length; i++) {
                ret[i] = (Double) get(i);
            }

            return (T[]) ret;
        }
        else if(number instanceof Float) {
            Float[] ret = new Float[size()];
            for(int i = 0; i < ret.length; i++) {
                ret[i] = (Float) get(i);
            }

            return (T[]) ret;
        }

        throw new UnsupportedOperationException("Unable to convert to array");
    }

    @Override
    public boolean add(X aX) {
        if(container == null) {
            container = Nd4j.create(10);
        }
        else if(size == container.length()) {
            growCapacity(size * 2);
        }
        if(DataTypeUtil.getDtypeFromContext() == DataType.DOUBLE)
            container.putScalar(size,aX.doubleValue());
        else {
            container.putScalar(size,aX.floatValue());

        }

        size++;
        return true;
    }

    @Override
    public boolean remove(Object o) {
        int idx = BooleanIndexing.firstIndex(container,new EqualsCondition((double) o)).getInt(0);
        if(idx < 0)
            return false;
        container.put(new INDArrayIndex[]{NDArrayIndex.interval(idx,container.length())},container.get(NDArrayIndex.interval(idx + 1,container.length())));
        container = container.reshape(1,size);
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
    public boolean addAll(Collection<? extends X> collection) {
        if(collection instanceof BaseNDArrayList) {
            BaseNDArrayList ndArrayList = (BaseNDArrayList) collection;
            ndArrayList.growCapacity(this.size() + collection.size());

        }
        else {
            for(X d : collection) {
                add(d);
            }
        }
        return true;
    }

    @Override
    public boolean addAll(int i, Collection<? extends X> collection) {

        for(X d : collection) {
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
    public X get(int i) {
        Number ret = container.getDouble(i);
        return (X) ret;
    }

    @Override
    public X set(int i, X aX) {
        if(DataTypeUtil.getDtypeFromContext() == DataType.DOUBLE)
            container.putScalar(i,aX.doubleValue());
        else {
            container.putScalar(i,aX.floatValue());

        }


        return aX;
    }

    @Override
    public void add(int i, X aX) {
        rangeCheck(i);
        growCapacity(i);
        moveForward(i);
        if(DataTypeUtil.getDtypeFromContext() == DataType.DOUBLE)
            container.putScalar(i,aX.doubleValue());
        else {
            container.putScalar(i,aX.floatValue());

        }

        size++;
    }

    @Override
    public X remove(int i) {
        rangeCheck(i);
        int numMoved = this.size - i - 1;
        if(numMoved > 0) {
            Number move = container.getDouble(i);
            moveBackward(i);
            size--;
            return (X) move;
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
    public ListIterator<X> listIterator() {
        return new NDArrayListIterator();
    }

    @Override
    public ListIterator<X> listIterator(int i) {
        return new NDArrayListIterator(i);
    }



    @Override
    public String toString() {
        return container.get(NDArrayIndex.interval(0,size)).toString();
    }

    private class NDArrayListIterator implements ListIterator<X> {
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
        public X next() {
            Number ret = get(curr);
            curr++;
            return (X) ret;
        }

        @Override
        public boolean hasPrevious() {
            return curr > 0;
        }

        @Override
        public X previous() {
            Number ret = get(curr - 1);
            curr--;
            return (X) ret;
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
        public void set(X aX) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void add(X aX) {
            throw new UnsupportedOperationException();
        }
    }



    private void growCapacity(int idx) {
        if(container == null) {
            container = Nd4j.create(10);
        }
        else if(idx >= container.length()) {
            val max = Math.max(container.length() * 2,idx);
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
        INDArrayIndex[] first = new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.interval(index  ,index  + numMoved)};
        INDArrayIndex[] getRange = new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.interval(index + 1 ,index + 1  + numMoved)};
        INDArray get = container.get(getRange);
        container.put(first,get);
    }

    private void moveForward(int index) {
        int numMoved = size - index - 1;
        INDArrayIndex[] getRange = new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.interval(index,index + numMoved)};
        INDArray get = container.get(getRange);
        INDArrayIndex[] first = new INDArrayIndex[] {NDArrayIndex.point(0), NDArrayIndex.interval(index + 1,index + 1 + get.length())};
        container.put(first,get);
    }

}
