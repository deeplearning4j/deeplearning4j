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

package org.nd4j.list.matrix;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.list.BaseNDArrayList;

import java.util.*;

/**
 * An {@link ArrayList} like implementation of {@link List}
 * using {@link INDArray} as the backing data structure.
 *
 * Creates an internal container of ndarray lists with a matrix shape.
 *
 * @author Adam Gibson
 */
public abstract  class MatrixBaseNDArrayList<X extends BaseNDArrayList> extends  AbstractList<X>  {
    private List<X> list = new ArrayList<>();



    /**
     * Get a view of the underlying array
     * relative to the size of the actual array.
     * (Sometimes there are overflows in the internals
     * but you want to use the internal INDArray for computing something
     * directly, this gives you the relevant subset that reflects the content of the list)
     * @return the view of the underlying ndarray relative to the collection's real size
     */
    public INDArray array() {
        List<INDArray> retList = new ArrayList<>(list.size());
        for(X x : list) {
            retList.add(x.array());
        }

        return Nd4j.concat(0,retList.toArray(new INDArray[retList.size()]));
    }

    @Override
    public int size() {
        return list.size();
    }

    @Override
    public boolean isEmpty() {
        return list.isEmpty();
    }

    @Override
    public boolean contains(Object o) {
        return list.contains(o);
    }

    @Override
    public Iterator<X> iterator() {
        return list.iterator();
    }

    @Override
    public Object[] toArray() {
        return list.toArray();
    }

    @Override
    public <T> T[] toArray(T[] ts) {
      return list.toArray(ts);
    }

    @Override
    public boolean add(X aX) {
        return list.add(aX);
    }

    @Override
    public boolean remove(Object o) {
        return list.remove(o);
    }

    @Override
    public boolean containsAll(Collection<?> collection) {
        return list.containsAll(collection);
    }

    @Override
    public boolean addAll(Collection<? extends X> collection) {
        return list.addAll(collection);
    }

    @Override
    public boolean addAll(int i, Collection<? extends X> collection) {
        return list.addAll(collection);
    }

    @Override
    public boolean removeAll(Collection<?> collection) {
        return list.removeAll(collection);
    }

    @Override
    public boolean retainAll(Collection<?> collection) {
        return list.retainAll(collection);
    }

    @Override
    public void clear() {
        list.clear();
    }

    @Override
    public X get(int i) {
        return list.get(i);
    }

    @Override
    public X set(int i, X aX) {
        return list.set(i,aX);
    }

    @Override
    public void add(int i, X aX) {
        list.add(i,aX);
    }

    @Override
    public X remove(int i) {
        return list.remove(i);
    }

    @Override
    public int indexOf(Object o) {
        return list.indexOf(o);
    }

    @Override
    public int lastIndexOf(Object o) {
        return list.lastIndexOf(o);
    }

    @Override
    public ListIterator<X> listIterator() {
        return list.listIterator();
    }

    @Override
    public ListIterator<X> listIterator(int i) {
        return list.listIterator(i);
    }



    @Override
    public String toString() {
        return list.toString();
    }


    /**
     * Get entry i,j in the matrix
     * @param i the row
     * @param j the column
     * @return the entry at i,j if it exists
     */
    public Number getEntry(int i,int j) {
        return list.get(i).get(j);
    }

}
