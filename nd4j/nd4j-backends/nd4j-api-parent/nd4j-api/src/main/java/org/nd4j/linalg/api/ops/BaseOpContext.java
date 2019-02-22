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

import lombok.val;
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

    protected List<Double> fastpath_d = new ArrayList<>();
    protected List<Boolean> fastpath_b = new ArrayList<>();
    protected List<Long> fastpath_i = new ArrayList<>();

    @Override
    public void setIArguments(long... arguments) {
        fastpath_i.clear();
        for (val v:arguments)
            fastpath_i.add(v);
    }

    @Override
    public void setTArguments(double... arguments) {
        fastpath_d.clear();
        for (val v:arguments)
            fastpath_d.add(v);
    }

    @Override
    public void setBArguments(boolean... arguments) {
        fastpath_b.clear();
        for (val v:arguments)
            fastpath_b.add(v);
    }

    @Override
    public void setInputArray(int index, INDArray array) {
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
    public void setOutputArray(int index, INDArray array) {
        fastpath_out.put(index, array);
    }


}
