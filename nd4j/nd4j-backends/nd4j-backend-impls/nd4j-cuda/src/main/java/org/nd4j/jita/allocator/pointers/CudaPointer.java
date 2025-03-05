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

package org.nd4j.jita.allocator.pointers;

import org.bytedeco.javacpp.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class is simple logic-less holder for pointers derived from CUDA.
 *
 * PLEASE NOTE:
 * 1. All pointers are blind, and do NOT care about length/capacity/offsets/strides whatever
 * 2. They are really blind. Even data opType is unknown.
 *
 * @author raver119@gmail.com
 */
public class CudaPointer extends Pointer {

    private static Logger logger = LoggerFactory.getLogger(CudaPointer.class);


    public CudaPointer(Pointer pointer) {
        this.address = pointer.address();
        this.capacity = pointer.capacity();
        this.limit = pointer.limit();
        this.position = pointer.position();
    }

    public CudaPointer(Pointer pointer, long capacity) {
        this.address = pointer.address();
        this.capacity = capacity;
        this.limit = capacity;
        this.position = 0;

    }

    public CudaPointer(Pointer pointer, long capacity, long byteOffset) {
        this.address = pointer.address() + byteOffset;
        this.capacity = capacity;
        this.limit = capacity;
        this.position = 0;
    }

    public CudaPointer(long address) {
        this.address = address;
    }

    public CudaPointer(long address, long capacity) {
        this.address = address;
        this.capacity = capacity;
        this.limit = capacity;
        this.position = 0;
    }

    public Pointer asNativePointer() {
        return new Pointer(this);
    }

    public FloatPointer asFloatPointer() {
        return new FloatPointer(this);
    }

    public LongPointer asLongPointer() {
        return new LongPointer(this);
    }

    public DoublePointer asDoublePointer() {
        return new DoublePointer(this);
    }

    public IntPointer asIntPointer() {
        return new IntPointer(this);
    }

    public ShortPointer asShortPointer() {
        return new ShortPointer(this);
    }

    public BytePointer asBytePointer() {
        return new BytePointer(this);
    }

    public BooleanPointer asBooleanPointer() {
        return new BooleanPointer(this);
    }

    public long getNativePointer() {
        return address();
    }

    /**
     * Returns 1 for Pointer or BytePointer else {@code Loader.sizeof(getClass())} or -1 on error.
     */
    @Override
    public int sizeof() {
        return 4;
    }
}
