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

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import java.nio.DoubleBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

@Namespace("sd::ops") @NoOffset
public class OpExecTrace extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OpExecTrace(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public OpExecTrace(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public OpExecTrace position(long position) {
        return (OpExecTrace)super.position(position);
    }
    @Override public OpExecTrace getPointer(long i) {
        return new OpExecTrace((Pointer)this).offsetAddress(i);
    }

    public native @Cast("const sd::LongType**") @StdVector PointerPointer inputShapeBuffers(); public native OpExecTrace inputShapeBuffers(PointerPointer setter);
    public native @Cast("const sd::LongType**") @StdVector PointerPointer outputShapeBuffers(); public native OpExecTrace outputShapeBuffers(PointerPointer setter);
    public native @StdString
    @Cast({"char*", "std::string*"}) BytePointer opName(); public native OpExecTrace opName(BytePointer setter);
    public native @StdVector IntPointer iArgs(); public native OpExecTrace iArgs(IntPointer setter);
    public native @StdVector DoublePointer tArgs(); public native OpExecTrace tArgs(DoublePointer setter);
    public native @Cast("sd::DataType*") @StdVector IntPointer dArgs(); public native OpExecTrace dArgs(IntPointer setter);
    public native @Cast("bool*") @StdVector BooleanPointer bArgs(); public native OpExecTrace bArgs(BooleanPointer setter);
    public native @ByRef org.nd4j.nativeblas.StringVector sArguments(); public native OpExecTrace sArguments(org.nd4j.nativeblas.StringVector setter);
    public native int opType(); public native OpExecTrace opType(int setter);


// #ifndef __JAVACPP_HACK__
// #endif

    public OpExecTrace() { super((Pointer)null); allocate(); }
    private native void allocate();

    public native @Cast("const sd::LongType**") @StdVector PointerPointer getInputShapeBuffers();
    public native void setInputShapeBuffers(@Cast("const sd::LongType**") @StdVector PointerPointer inputShapeBuffers);
    public native void setInputShapeBuffers(@Cast("const sd::LongType**") @StdVector @ByPtrPtr LongPointer inputShapeBuffers);
    public native void setInputShapeBuffers(@Cast("const sd::LongType**") @StdVector @ByPtrPtr LongBuffer inputShapeBuffers);
    public native void setInputShapeBuffers(@Cast("const sd::LongType**") @StdVector @ByPtrPtr long[] inputShapeBuffers);
    public native @Cast("const sd::LongType**") @StdVector PointerPointer getOutputShapeBuffers();
    public native void setOutputShapeBuffers(@Cast("const sd::LongType**") @StdVector PointerPointer outputShapeBuffers);
    public native void setOutputShapeBuffers(@Cast("const sd::LongType**") @StdVector @ByPtrPtr LongPointer outputShapeBuffers);
    public native void setOutputShapeBuffers(@Cast("const sd::LongType**") @StdVector @ByPtrPtr LongBuffer outputShapeBuffers);
    public native void setOutputShapeBuffers(@Cast("const sd::LongType**") @StdVector @ByPtrPtr long[] outputShapeBuffers);
    public native @StdString @Cast({"char*", "std::string*"}) BytePointer getOpName();
    public native void setOpName(@StdString @Cast({"char*", "std::string*"}) BytePointer opName);
    public native @StdVector IntPointer getIArgs();
    public native void setIArgs(@StdVector IntPointer iArgs);
    public native void setIArgs(@StdVector IntBuffer iArgs);
    public native void setIArgs(@StdVector int[] iArgs);
    public native @StdVector DoublePointer getTArgs();
    public native void setTArgs(@StdVector DoublePointer tArgs);
    public native void setTArgs(@StdVector DoubleBuffer tArgs);
    public native void setTArgs(@StdVector double[] tArgs);
    public native @Cast("sd::DataType*") @StdVector IntPointer getDArgs();
    public native void setDArgs(@Cast("sd::DataType*") @StdVector IntPointer dArgs);
    public native void setDArgs(@Cast("sd::DataType*") @StdVector IntBuffer dArgs);
    public native void setDArgs(@Cast("sd::DataType*") @StdVector int[] dArgs);
    public native @Cast("bool*") @StdVector BooleanPointer getBArgs();
    public native void setBArgs(@Cast("bool*") @StdVector BooleanPointer bArgs);
    public native void setBArgs(@Cast("bool*") @StdVector boolean[] bArgs);
    public native @Const @ByRef org.nd4j.nativeblas.StringVector getSArguments();
    public native void setSArguments(@Const @ByRef org.nd4j.nativeblas.StringVector sArguments);
    public native int getOpType();
    public native void setOpType(int opType);
}