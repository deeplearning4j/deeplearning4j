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

package org.nd4j.linalg.api.ops.impl.broadcast;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseBroadcastOp;

import java.util.List;

public class BroadcastGradientArgs extends BaseBroadcastOp {
    public BroadcastGradientArgs(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, int[] dimension) {
        super(sameDiff, i_v1, i_v2, dimension);
    }

    public BroadcastGradientArgs(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace, int[] dimension) {
        super(sameDiff, i_v1, i_v2, inPlace, dimension);
    }

    public BroadcastGradientArgs(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, int[] dimension, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, dimension, extraArgs);
    }

    public BroadcastGradientArgs() {}

    public BroadcastGradientArgs(INDArray x, INDArray y, INDArray z, int... dimension) {
        super(x, y, z, dimension);
    }



    @Override
    public int opNum() {
        return 0;
    }

    @Override
    public String opName() {
        return "BroadcastGradientArgs".toLowerCase();
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }

    @Override
    public String onnxName() {
       throw new NoOpNameFoundException("No op name found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "BroadcastGradientArgs";
    }
}
