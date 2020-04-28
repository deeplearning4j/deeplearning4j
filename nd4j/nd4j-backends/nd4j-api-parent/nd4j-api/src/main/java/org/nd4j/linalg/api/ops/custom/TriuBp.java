/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.nd4j.linalg.api.ops.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.ReverseBp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

@NoArgsConstructor
public class TriuBp extends DynamicCustomOp {

    public TriuBp(SameDiff sameDiff, SDVariable in, SDVariable grad, int diag) {
        super(sameDiff, new SDVariable[]{in, grad});
        addIArgument(diag);
    }

    public TriuBp(SameDiff sameDiff, SDVariable in, SDVariable grad) {
        super(sameDiff, new SDVariable[]{in, grad});
    }

    @Override
    public String opName() {
        return "triu_bp";
    }


    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {

        return Collections.singletonList(arg(0).dataType());
    }


}
