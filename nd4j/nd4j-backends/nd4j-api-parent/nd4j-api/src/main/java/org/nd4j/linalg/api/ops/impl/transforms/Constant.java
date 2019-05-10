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

package org.nd4j.linalg.api.ops.impl.transforms;

import lombok.Data;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.BaseTransformSameOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

@Data
public class Constant extends BaseTransformSameOp {


    public Constant() {
    }


    protected Constant(SameDiff sameDiff,
                       SDVariable i_v,
                       long[] shape,
                       boolean inPlace) {
        super();
        sameDiff.putOrUpdateShapeForVarName(i_v.getVarName(), shape, false);
        this.xVertexId = i_v.getVarName();
        this.inPlace = inPlace;
        this.sameDiff = sameDiff;
    }

    public Constant(SameDiff sameDiff, SDVariable i_v, long[] shape) {
        this(sameDiff, i_v, shape, false);
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }


    @Override
    public DifferentialFunction dup() {
        Constant ret = new Constant(sameDiff, sameDiff.getVariable(outputVariables()[0].getVarName())
                , sameDiff.getShapeForVarName(outputVariables()[0].getVarName()));
        Constant differentialFunction = ret;
        return differentialFunction;
    }


    @Override
    public int opNum() {
        return 15;
    }

    @Override
    public String opName() {
        return "constant";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow opName found for " + opName());
    }

}
