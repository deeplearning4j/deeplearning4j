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

package org.nd4j.linalg.api.ops;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.val;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

/**
 * Implementation of common methods for OpContext
 *
 * @author raver119@gmail.com
 */
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

    @Override
    public void setIArguments(long... arguments) {
        fastpath_i.clear();
        for (val v:arguments)
            fastpath_i.add(v);
    }

    @Override
    public List<Long> getIArguments(){
        return fastpath_i;
    }

    @Override
    public void setTArguments(double... arguments) {
        fastpath_t.clear();
        for (val v:arguments)
            fastpath_t.add(v);
    }

    @Override
    public List<Double> getTArguments(){
        return fastpath_t;
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
    public void setDArguments(DataType... arguments) {
        fastpath_d.clear();
        for (val v:arguments)
            fastpath_d.add(v);
    }

    @Override
    public List<DataType> getDArguments() {
        return fastpath_d;
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
}
