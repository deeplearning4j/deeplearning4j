/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.List;

@NoArgsConstructor
public class MaximumBp extends DynamicCustomOp {

    public MaximumBp(@NonNull SameDiff sameDiff, @NonNull SDVariable x, @NonNull SDVariable y,  @NonNull SDVariable gradO) {
        super("maximum_bp",sameDiff, new SDVariable[]{x,y, gradO});
    }

    @Override
    public String opName() {
        return "maximum_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        List<DataType> list = new ArrayList<DataType>();
        list.add(inputDataTypes.get(0));
        list.add(inputDataTypes.get(0));
        return list;
    }
}
