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

package org.nd4j.linalg.api.ops.impl.reduce.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * Backprop op for Norm Max reduction operation
 *
 * @author Alex Black
 */

public class NormMaxBp extends BaseReductionBp {

    public NormMaxBp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean keepDims, int... dimensions) {
        super(sameDiff, origInput, gradAtOutput, keepDims, dimensions);
    }

    public NormMaxBp(INDArray origInput, INDArray gradAtOutput, INDArray output, boolean keepDims, int... dimensions){
        super(origInput, gradAtOutput, output, keepDims, dimensions);
    }

    public NormMaxBp(){}

    @Override
    public String opName() {
        return "reduce_norm_max_bp";
    }
}
