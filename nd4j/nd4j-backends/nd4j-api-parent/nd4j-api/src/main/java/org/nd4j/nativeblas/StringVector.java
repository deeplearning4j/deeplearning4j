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
package org.nd4j.nativeblas;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;

@Name("std::vector<std::string>") public class StringVector extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public StringVector(Pointer p) { super(p); }
    public StringVector(BytePointer value) { this(1); put(0, value); }
    public StringVector(BytePointer ... array) { this(array.length); put(array); }
    public StringVector(String value) { this(1); put(0, value); }
    public StringVector(String ... array) { this(array.length); put(array); }
    public StringVector()       { allocate();  }
    public StringVector(long n) { allocate(n); }
    private native void allocate();
    private native void allocate(@Cast("size_t") long n);
    public native @Name("operator =") @ByRef StringVector put(@ByRef StringVector x);

    public boolean empty() { return size() == 0; }
    public native long size();
    public void clear() { resize(0); }
    public native void resize(@Cast("size_t") long n);

    @Index(function = "at") public native @StdString BytePointer get(@Cast("size_t") long i);
    public native StringVector put(@Cast("size_t") long i, BytePointer value);
    @ValueSetter @Index(function = "at") public native StringVector put(@Cast("size_t") long i, @StdString String value);

    public native @ByVal Iterator insert(@ByVal Iterator pos, @StdString BytePointer value);
    public native @ByVal Iterator erase(@ByVal Iterator pos);
    public native @ByVal Iterator begin();
    public native @ByVal Iterator end();
    @NoOffset @Name("iterator") public static class Iterator extends Pointer {
        public Iterator(Pointer p) { super(p); }
        public Iterator() { }

        public native @Name("operator ++") @ByRef Iterator increment();
        public native @Name("operator ==") boolean equals(@ByRef Iterator it);
        public native @Name("operator *") @StdString BytePointer get();
    }

    public BytePointer[] get() {
        BytePointer[] array = new BytePointer[size() < Integer.MAX_VALUE ? (int)size() : Integer.MAX_VALUE];
        for (int i = 0; i < array.length; i++) {
            array[i] = get(i);
        }
        return array;
    }
    @Override public String toString() {
        return java.util.Arrays.toString(get());
    }

    public BytePointer pop_back() {
        long size = size();
        BytePointer value = get(size - 1);
        resize(size - 1);
        return value;
    }
    public StringVector push_back(BytePointer value) {
        long size = size();
        resize(size + 1);
        return put(size, value);
    }
    public StringVector put(BytePointer value) {
        if (size() != 1) { resize(1); }
        return put(0, value);
    }
    public StringVector put(BytePointer ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }

    public StringVector push_back(String value) {
        long size = size();
        resize(size + 1);
        return put(size, value);
    }
    public StringVector put(String value) {
        if (size() != 1) { resize(1); }
        return put(0, value);
    }
    public StringVector put(String ... array) {
        if (size() != array.length) { resize(array.length); }
        for (int i = 0; i < array.length; i++) {
            put(i, array[i]);
        }
        return this;
    }
}