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

package org.nd4j.linalg.api.ops;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.val;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.OpaqueNDArray;
import org.nd4j.shade.guava.primitives.Booleans;
import org.nd4j.shade.guava.primitives.Doubles;
import org.nd4j.shade.guava.primitives.Longs;

import java.util.*;

public abstract class BaseOpContext implements OpContext {
    protected Map<Integer,INDArray> fastpath_in = new HashMap<>();
    protected Map<Integer,INDArray> fastpath_out = new HashMap<>();

    protected List<Double> fastpath_t = new ArrayList<>();
    protected List<Boolean> fastpath_b = new ArrayList<>();
    protected List<Long> fastpath_i = new ArrayList<>();

    protected List<DataType> fastpath_d = new ArrayList<>();

    @Setter()
    @Getter
    protected ExecutionMode executionMode = ExecutionMode.UNDEFINED;

    protected volatile boolean closed = false;

    @Override
    public void setArgsFrom(CustomOp customOp) {
        setIArguments(customOp.iArgs());
        setTArguments(customOp.tArgs());
        setBArguments(customOp.bArgs());
        setDArguments(customOp.dArgs());
        setInputArrays(customOp.inputArguments());
        setOutputArrays(customOp.outputArguments());
    }


    @Override
    public boolean bArgumentAtNative(int index) {
        return false;
    }

    @Override
    public INDArray getInputArrayNative(int idx) {
        return OpaqueNDArray.toINDArray(Nd4j.getNativeOps().getInputArrayNative(contextPointer(), idx));
    }

    @Override
    public INDArray getOutputArrayNative(int idx) {
        return OpaqueNDArray.toINDArray(Nd4j.getNativeOps().getOutputArrayNative(contextPointer(), idx));
    }

    @Override
    public DataType dataTypeNativeAt(int index) {
        return DataType.fromInt((int) Nd4j.getNativeOps().dataTypeNativeAt(contextPointer(), index));
    }

    @Override
    public int iArgumentAtNative(int index) {
        return (int) Nd4j.getNativeOps().iArgumentAtNative(contextPointer(), index);
    }

    @Override
    public int numDNative() {
        return (int) Nd4j.getNativeOps().numDNative(contextPointer());
    }

    @Override
    public int numInputsNative() {
        return (int) Nd4j.getNativeOps().numInputsNative(contextPointer());
    }

    @Override
    public int numTArgumentsNative() {
        return (int) Nd4j.getNativeOps().numTArgumentsNative(contextPointer());
    }

    @Override
    public Double tArgumentNative(int index) {
        return Nd4j.getNativeOps().tArgumentNative(contextPointer(), index);
    }

    @Override
    public int numOutArgumentsNative() {
        return (int) Nd4j.getNativeOps().numOutputsNative(contextPointer());
    }


    @Override
    public int numIArgumentsNative() {
        return (int) Nd4j.getNativeOps().numIArgumentsNative(contextPointer());
    }
    @Override
    public int numBArgumentsNative() {
        return (int) Nd4j.getNativeOps().numBNative(contextPointer());
    }

    @Override
    public void setIArguments(Pointer arguments, int length) {
        throw new UnsupportedOperationException("Unable to set an int arguments pointer using a pointer");

    }

    @Override
    public void setTArguments(Pointer arguments, int length) {
        throw new UnsupportedOperationException("Unable to set an double arguments pointer using a pointer");

    }

    @Override
    public void setDArguments(Pointer arguments, int length) {
        throw new UnsupportedOperationException("Unable to set a data type arguments pointer using a pointer");

    }

    @Override
    public void setBArguments(Pointer arguments, int length) {
        throw new UnsupportedOperationException("Unable to set a boolean arguments pointer using a pointer");

    }

    @Override
    public void setBArguments(List<Boolean> arguments) {
        setBArguments(Booleans.toArray(arguments));
    }


    @Override
    public void setIArguments(long... arguments) {
        fastpath_i.clear();
        for (val v:arguments)
            fastpath_i.add(v);
    }

    @Override
    public void setIArguments(List<Long> iArguments) {
        setIArguments(Longs.toArray(iArguments));
    }

    @Override
    public List<Long> getIArguments(){
        return fastpath_i;
    }

    @Override
    public int numIArguments() {
        return fastpath_i.size();
    }

    @Override
    public void setTArguments(double... arguments) {
        fastpath_t.clear();
        for (val v:arguments)
            fastpath_t.add(v);
    }

    @Override
    public void setTArguments(List<Double> tArguments) {
        setTArguments(Doubles.toArray(tArguments));
    }

    @Override
    public List<Double> getTArguments(){
        return fastpath_t;
    }

    @Override
    public int numTArguments() {
        return fastpath_t.size();
    }

    @Override
    public void setBArguments(boolean... arguments) {
        fastpath_b.clear();
        for (val v:arguments)
            fastpath_b.add(v);
    }



    @Override
    public List<Boolean> getBArguments(){
        return fastpath_b;
    }

    @Override
    public int numBArguments() {
        return fastpath_b.size();
    }

    @Override
    public void setDArguments(DataType... arguments) {
        fastpath_d.clear();
        for (val v:arguments)
            fastpath_d.add(v);
    }

    @Override
    public void setDArguments(List<DataType> arguments) {
        setDArguments(arguments.toArray(new DataType[0]));
    }

    @Override
    public List<DataType> getDArguments() {
        return fastpath_d;
    }

    @Override
    public int numDArguments() {
        return fastpath_d.size();
    }

    @Override
    public void setInputArray(int index, @NonNull INDArray array) {
        fastpath_in.put(index, array);
    }

    @Override
    public List<INDArray> getInputArrays() {
        val result = new ArrayList<INDArray>();
        for (int e = 0; e < Integer.MAX_VALUE; e++) {
            val arr = fastpath_in.get(e);
            if (arr != null)
                result.add(arr);
            else
                break;
        }

        return result;
    }

    @Override
    public int numInputArguments() {
        return fastpath_in.size();
    }

    @Override
    public INDArray getInputArray(int idx) {
        return fastpath_in.get(idx);
    }

    @Override
    public List<INDArray> getOutputArrays() {
        val result = new ArrayList<INDArray>();
        for (int e = 0; e < Integer.MAX_VALUE; e++) {
            val arr = fastpath_out.get(e);
            if (arr != null)
                result.add(arr);
            else
                break;
        }

        return result;
    }

    @Override
    public void setOutputArray(int index, @NonNull INDArray array) {
        fastpath_out.put(index, array);
    }

    @Override
    public INDArray getOutputArray(int i) {
        return fastpath_out.get(i);
    }

    @Override
    public int numOutputArguments() {
        return fastpath_out.size();
    }

    @Override
    public void setInputArrays(@NonNull List<INDArray> arrays) {
        for (int e = 0; e < arrays.size(); e++)
            setInputArray(e, arrays.get(e));
    }

    @Override
    public void setOutputArrays(@NonNull List<INDArray> arrays) {
        for (int e = 0; e < arrays.size(); e++)
            setOutputArray(e, arrays.get(e));
    }

    @Override
    public void setInputArrays(INDArray... arrays) {
        for (int e = 0; e < arrays.length; e++)
            setInputArray(e, arrays[e]);
    }

    @Override
    public void setOutputArrays(INDArray... arrays) {
        for (int e = 0; e < arrays.length; e++)
            setOutputArray(e, arrays[e]);
    }

    @Override
    public void purge() {
        fastpath_in.clear();
        fastpath_out.clear();
    }

    @Override
    public void setArgs(INDArray[] inputArrs, long[] iArgs, DataType[] dArgs, double[] tArgs, boolean[] bArgs) {
        if (inputArrs != null) {
            setInputArrays(inputArrs);
        }
        if (iArgs != null)
            setIArguments(iArgs);
        if (dArgs != null)
            setDArguments(dArgs);
        if (tArgs != null)
            setTArguments(tArgs);
        if (bArgs != null)
            setBArguments(bArgs);
    }

    @Override
    public void transferTArgs() {
        setTArguments();
    }

    @Override
    public void transferIArgs() {

    }

    @Override
    public void transferBArgs() {

    }

    @Override
    public void transferDArgs() {

    }

    public boolean isClosed() {
        return closed;
    }

    protected void setClosed(boolean closed) {
        this.closed = closed;
    }
}
