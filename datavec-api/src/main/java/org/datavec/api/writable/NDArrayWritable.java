/*-
 *  * Copyright 2017 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */
package org.datavec.api.writable;

import org.datavec.api.io.WritableComparable;
import org.datavec.api.util.ndarray.DataInputWrapperStream;
import org.datavec.api.util.ndarray.DataOutputWrapperStream;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.util.MathUtils;

import java.io.*;
import java.util.Arrays;

/**
 * A Writable that basically wraps an INDArray.
 *
 * @author saudet
 */
public class NDArrayWritable extends ArrayWritable implements WritableComparable {
    public static final byte NDARRAY_SER_VERSION_HEADER_NULL = 0;
    public static final byte NDARRAY_SER_VERSION_HEADER = 1;

    private INDArray array = null;
    private Integer hash = null;

    public NDArrayWritable() {}

    public NDArrayWritable(INDArray array) {
        set(array);
    }

    /**
     * Deserialize into a row vector of default type.
     */
    public void readFields(DataInput in) throws IOException {
        DataInputStream dis = new DataInputStream(new DataInputWrapperStream(in));
        byte header = dis.readByte();
        if (header != NDARRAY_SER_VERSION_HEADER && header != NDARRAY_SER_VERSION_HEADER_NULL) {
            throw new IllegalStateException("Unexpected NDArrayWritable version header - stream corrupt?");
        }

        if (header == NDARRAY_SER_VERSION_HEADER_NULL) {
            array = null;
            return;
        }

        array = Nd4j.read(dis);
        hash = null;
    }

    @Override
    public void writeType(DataOutput out) throws IOException {
        out.writeShort(WritableType.NDArray.typeIdx());
    }

    @Override
    public WritableType getType() {
        return WritableType.NDArray;
    }

    /**
     * Serialize array data linearly.
     */
    public void write(DataOutput out) throws IOException {
        if (array == null) {
            out.write(NDARRAY_SER_VERSION_HEADER_NULL);
            return;
        }

        INDArray toWrite;
        if (array.isView()) {
            toWrite = array.dup();
        } else {
            toWrite = array;
        }

        //Write version header: this allows us to maintain backward compatibility in the future,
        // with features such as compression, sparse arrays or changes on the DataVec side
        out.write(NDARRAY_SER_VERSION_HEADER);
        Nd4j.write(toWrite, new DataOutputStream(new DataOutputWrapperStream(out)));
    }

    public void set(INDArray array) {
        this.array = array;
        this.hash = null;
    }

    public INDArray get() {
        return array;
    }

    /**
     * Returns true iff <code>o</code> is a ArrayWritable with the same value.
     */
    public boolean equals(Object o) {
        if (!(o instanceof NDArrayWritable)) {
            return false;
        }
        INDArray io = ((NDArrayWritable) o).get();

        if (this.array == null && io != null || this.array != null && io == null) {
            return false;
        }

        if (this.array == null) {
            //Both are null
            return true;
        }

        //For NDArrayWritable: we use strict equality. Otherwise, we can have a.equals(b) but a.hashCode() != b.hashCode()
        return this.array.equalsWithEps(io, 0.0);
    }

    @Override
    public int hashCode() {
        if (hash != null) {
            return hash;
        }

        //Hashcode needs to be invariant to array order - otherwise, equal arrays can have different hash codes
        // for example, C vs. F order arrays with otherwise identical contents

        if (array == null) {
            hash = 0;
            return hash;
        }

        int hash = Arrays.hashCode(array.shape());
        int length = array.length();
        NdIndexIterator iter = new NdIndexIterator('c', array.shape());
        for (int i = 0; i < length; i++) {
            hash ^= MathUtils.hashCode(array.getDouble(iter.next()));
        }

        this.hash = hash;
        return hash;
    }

    @Override
    public int compareTo(@NotNull Object o) {
        NDArrayWritable other = (NDArrayWritable) o;

        //Conventions used here for ordering NDArrays: x.compareTo(y): -ve if x < y, 0 if x == y, +ve if x > y
        //Null first
        //Then smallest rank first
        //Then smallest length first
        //Then sort by shape
        //Then sort by contents
        //The idea: avoid comparing contents for as long as possible

        if (this.array == null) {
            if (other.array == null) {
                return 0;
            }
            return -1;
        }
        if (other.array == null) {
            return 1;
        }

        if (this.array.rank() != other.array.rank()) {
            return Integer.compare(array.rank(), other.array.rank());
        }

        if (array.length() != other.array.length()) {
            return Long.compare(array.length(), other.array.length());
        }

        for (int i = 0; i < array.rank(); i++) {
            if (Long.compare(array.size(i), other.array.size(i)) != 0) {
                return Long.compare(array.size(i), other.array.size(i));
            }
        }

        //At this point: same rank, length, shape
        NdIndexIterator iter = new NdIndexIterator('c', array.shape());
        while (iter.hasNext()) {
            long[] nextPos = iter.next();
            double d1 = array.getDouble(nextPos);
            double d2 = other.array.getDouble(nextPos);

            if (Double.compare(d1, d2) != 0) {
                return Double.compare(d1, d2);
            }
        }

        //Same rank, length, shape and contents: must be equal
        return 0;
    }

    public String toString() {
        return array.toString();
    }

    @Override
    public long length() {
        return array.data().length();
    }

    @Override
    public double getDouble(long i) {
        return array.data().getDouble(i);
    }

    @Override
    public float getFloat(long i) {
        return array.data().getFloat(i);
    }

    @Override
    public int getInt(long i) {
        return array.data().getInt(i);
    }

    @Override
    public long getLong(long i) {
        return (long) array.data().getDouble(i);
    }
}
