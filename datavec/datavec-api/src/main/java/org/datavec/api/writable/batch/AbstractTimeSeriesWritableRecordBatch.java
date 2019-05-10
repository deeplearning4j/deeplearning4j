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

package org.datavec.api.writable.batch;

import org.datavec.api.writable.Writable;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

/**
 * Simple base class for List<List<Writable>></Writable>
 * implementations
 *
 * @author Alex Black
 */
public abstract class AbstractTimeSeriesWritableRecordBatch implements List<List<List<Writable>>> {


    @Override
    public boolean isEmpty() {
        return size() == 0;
    }

    @Override
    public boolean contains(Object o) {
        return false;
    }

    @Override
    public Iterator<List<List<Writable>>> iterator() {
        return listIterator();
    }

    @Override
    public ListIterator<List<List<Writable>>> listIterator() {
        throw new UnsupportedOperationException();
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
    public boolean add(List<List<Writable>> writable) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean remove(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean containsAll(Collection<?> collection) {
        return false;
    }

    @Override
    public boolean addAll(Collection<? extends List<List<Writable>>> collection) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean addAll(int i,  Collection<? extends List<List<Writable>>> collection) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean removeAll(Collection<?> collection) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean retainAll(Collection<?> collection) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void clear() {

    }

    @Override
    public List<List<Writable>> set(int i, List<List<Writable>> writable) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void add(int i, List<List<Writable>> writable) {
        throw new UnsupportedOperationException();

    }

    @Override
    public List<List<Writable>> remove(int i) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int indexOf(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int lastIndexOf(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override
    public ListIterator<List<List<Writable>>> listIterator(int i) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<List<List<Writable>>> subList(int i, int i1) {
        throw new UnsupportedOperationException();
    }



}
