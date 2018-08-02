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

package org.nd4j.linalg.api.ops.impl.transforms.gradient;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;

/**
 * Softmax backpropagation op - dL/dIn from in and dL/dOut
 *
 * @author Alex Black
 */
public class SoftmaxBp extends DynamicCustomOp {

    public SoftmaxBp(){ }

    public SoftmaxBp(SameDiff sd, SDVariable input, SDVariable grad, Integer dimension){
        super(null, sd, new SDVariable[]{input, grad});
        if(dimension != null)
            addIArgument(dimension);
    }

    @Override
    public String opName() {
        return "softmax_bp";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        throw new UnsupportedOperationException("Differentiating op softmax_bp not supported");
    }

}
