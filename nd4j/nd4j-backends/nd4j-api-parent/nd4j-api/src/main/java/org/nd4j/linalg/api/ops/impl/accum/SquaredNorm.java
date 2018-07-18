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

package org.nd4j.linalg.api.ops.impl.accum;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;


/**
 * Squared norm (sum_i x_i^2) reduction operation
 *
 * @author Alex Black
 */

public class SquaredNorm extends BaseReduction {
    public SquaredNorm(SameDiff sameDiff, SDVariable input, boolean keepDims, int... dimensions) {
        super(sameDiff, input, keepDims, dimensions);
    }

    public SquaredNorm(INDArray input, INDArray output, boolean keepDims, int... dimensions){
        super(input, output, keepDims, dimensions);
    }

    public SquaredNorm(){}


    @Override
    public String opName() {
        return "reduce_sqnorm";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        return Collections.singletonList(f().squaredNormBp(arg(), grad.get(0), keepDims, dimensions));
    }
}
