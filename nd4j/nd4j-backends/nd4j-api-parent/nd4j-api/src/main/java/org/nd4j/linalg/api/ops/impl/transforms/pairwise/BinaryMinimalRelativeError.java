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

package org.nd4j.linalg.api.ops.impl.transforms.pairwise;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;

import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class BinaryMinimalRelativeError extends BaseTransformOp {
    private double thresholdRelative = 0.0;
    private double thresholdAbsolute = 0.0;

    public BinaryMinimalRelativeError(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public BinaryMinimalRelativeError(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public BinaryMinimalRelativeError(SameDiff sameDiff) {
        super(sameDiff);
    }

    public BinaryMinimalRelativeError(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, extraArgs);
    }

    public BinaryMinimalRelativeError(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public BinaryMinimalRelativeError(SameDiff sameDiff, SDVariable i_v, int[] shape, boolean inPlace, Object[] extraArgs) {
        super(sameDiff, i_v, shape, inPlace, extraArgs);
    }

    public BinaryMinimalRelativeError(SameDiff sameDiff, SDVariable i_v, Object[] extraArgs) {
        super(sameDiff, i_v, extraArgs);
    }

    public BinaryMinimalRelativeError() {
    }

    public BinaryMinimalRelativeError(INDArray x, INDArray y, INDArray z, double thresholdRelative, double thresholdAbsolute, long n) {
        super(x, y, z, n);
        this.thresholdRelative = thresholdRelative;
        this.thresholdAbsolute = thresholdAbsolute;
    }

    public BinaryMinimalRelativeError(INDArray x, INDArray y, double thresholdRelative, double thresholdAbsolute) {
        super(x, y, x, x.lengthLong());
        this.thresholdRelative = thresholdRelative;
        this.thresholdAbsolute = thresholdAbsolute;
    }

    public BinaryMinimalRelativeError(INDArray x, INDArray y, INDArray z, double thresholdRelative, double thresholdAbsolute) {
        super(x, y, z, x.lengthLong());
        this.thresholdRelative = thresholdRelative;
        this.thresholdAbsolute = thresholdAbsolute;
    }

    @Override
    public int opNum() {
        return 28;
    }

    @Override
    public String opName() {
        return "BinaryMinimalRelativeError";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No  onnx opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No  onnx opName found for " + opName());
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[] {thresholdRelative, thresholdAbsolute};
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v1) {
        throw new UnsupportedOperationException();
    }
}
