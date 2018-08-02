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

package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Linspace/arange Op implementation, generates from..to distribution within Z
 *
 * @author raver119@gmail.com
 */
public class Linspace extends BaseRandomOp {
    private double from;
    private double to;
    private long length;

    public Linspace() {
        // no-op
    }

    public Linspace(double from, double to, int length) {
        this(Nd4j.createUninitialized(new int[] {1, length}, Nd4j.order()), from, to);
    }

    public Linspace(@NonNull INDArray z, double from, double to) {
        this.from = from;
        this.to = to;
        init(null, null, z, z.lengthLong());
        this.extraArgs = new Object[] {from, to};
    }

    public Linspace(SameDiff sd, double from, double to, long length){
        super(sd, new long[]{length});
        this.sameDiff = sd;
        this.from = from;
        this.to = to;
        this.length = length;
        this.extraArgs = new Object[] {from, to};
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("from",from);
        ret.put("to",to);
        return ret;
    }


    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName() {
        return "linspace";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    public List<long[]> calculateOutputShape() {
        return Collections.singletonList(new long[]{length});
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //No inputs
        return Collections.emptyList();
    }
}
