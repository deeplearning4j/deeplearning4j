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

package org.nd4j.linalg.api.ops.impl.transforms.comparison;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.BaseDynamicTransformOp;

import java.util.Arrays;
import java.util.List;

/**
 * Bit mask over the ndarrays as to whether
 * the components are greater than or not
 *
 * @author Adam Gibson
 */
public class GreaterThan extends BaseDynamicTransformOp {
    public GreaterThan() {}

    public GreaterThan( SameDiff sameDiff, SDVariable[] args, boolean inPlace) {
        super(sameDiff, args, inPlace);
    }

    public GreaterThan( INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }



    @Override
    public String opName() {
        return "greater";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        return "Greater";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        //2 inputs, not continuously differentiable but 0s almost everywhere
        return Arrays.asList(sameDiff.zerosLike(args()[0]), sameDiff.zerosLike(args()[1]));
    }
}
