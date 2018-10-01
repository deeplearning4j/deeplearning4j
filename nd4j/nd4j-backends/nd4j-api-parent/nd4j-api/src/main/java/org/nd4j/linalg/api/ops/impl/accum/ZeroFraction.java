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

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Compute the fraction of zero elements
 *
 * @author Max Pumperla
 */
@NoArgsConstructor
public class ZeroFraction extends DynamicCustomOp {

    public ZeroFraction(SameDiff sameDiff, SDVariable input) {

        super(null, sameDiff, new SDVariable[] {input}, false);
    }

    @Override
    public String opName() {
        return "zero_fraction";
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tf name found for shape " + opName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

}
