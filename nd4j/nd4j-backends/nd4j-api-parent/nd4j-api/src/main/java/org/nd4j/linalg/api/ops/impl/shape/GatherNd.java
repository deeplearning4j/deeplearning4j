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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * GatherND op
 */
@NoArgsConstructor
public class GatherNd extends DynamicCustomOp {


    public GatherNd(SameDiff sameDiff, SDVariable input, SDVariable indices, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input, indices}, inPlace);
    }

    @Override
    public String opName() {
        return "gather_nd";
    }

    @Override
    public String onnxName() {
        return "GatherND";
    }


    @Override
    public String[] tensorflowNames() {
        return new String[]{"GatherNd"};
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        //Output type is same as (first) input type
        return Collections.singletonList(dataTypes.get(0));
    }
}
