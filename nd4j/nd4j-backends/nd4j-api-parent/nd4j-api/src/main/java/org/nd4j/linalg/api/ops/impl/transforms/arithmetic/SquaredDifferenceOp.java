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

package org.nd4j.linalg.api.ops.impl.transforms.arithmetic;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.bp.SquaredDifferenceBpOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Squared difference operation, i.e. returns (x - y) * (x - y)
 *
 * @author Max Pumperla
 */
public class SquaredDifferenceOp extends BaseDynamicTransformOp {
    public static final String OP_NAME = "squaredsubtract";

    public SquaredDifferenceOp() {}

    public SquaredDifferenceOp(SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(sameDiff, args, inPlace);
    }

    public SquaredDifferenceOp(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }


    @Override
    public String opName() {
        return OP_NAME;
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "SquaredDifference";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        SDVariable[] outputs = new SquaredDifferenceBpOp(f().sameDiff(), new SDVariable[]{larg(), rarg(), i_v1.get(0)}).outputVariables();
        return Arrays.asList(outputs);
    }

}
