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
import org.nd4j.linalg.api.ops.BaseTransformBoolOp;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.Arrays;
import java.util.List;

/**
 * Bit mask over the ndarrays as to whether
 * the components are greater than or not
 *
 * @author Adam Gibson
 */
public class OldGreaterThan extends BaseTransformBoolOp {
    public OldGreaterThan(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public OldGreaterThan(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public OldGreaterThan(SameDiff sameDiff) {
        super(sameDiff);
    }

    public OldGreaterThan(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, extraArgs);
    }

    public OldGreaterThan(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public OldGreaterThan(SameDiff sameDiff, SDVariable i_v, long[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public OldGreaterThan(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public OldGreaterThan() {}

    public OldGreaterThan(INDArray x, INDArray z) {
        super(x, z);
    }

    public OldGreaterThan(INDArray x, INDArray z, long n) {
        super(x, z, n);
    }

    public OldGreaterThan(INDArray x, INDArray y, INDArray z, long n) {
        super(x, y, z, n);
    }

    public OldGreaterThan(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 1;
    }

    @Override
    public String opName() {
        return "oldgt";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow name found");
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Arrays.asList(outputVariables()[0]);
    }
}
