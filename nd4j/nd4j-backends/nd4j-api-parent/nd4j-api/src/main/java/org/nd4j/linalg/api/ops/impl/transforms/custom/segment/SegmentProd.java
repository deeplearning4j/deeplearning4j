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

package org.nd4j.linalg.api.ops.impl.transforms.custom.segment;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Segment product operation
 *
 * @author Alex Black
 */
public class SegmentProd extends DynamicCustomOp {

    public SegmentProd(SameDiff sameDiff, SDVariable data, SDVariable segmentIds) {
        super(null, sameDiff,  new SDVariable[] {data, segmentIds}, false);
    }

    public SegmentProd(){ }

    @Override
    public String opName(){
        return "segment_prod";
    }

    @Override
    public String tensorflowName() {
        return "SegmentProd";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Arrays.asList(f().segmentProdBp(arg(0), arg(1), gradients.get(0)));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 2, "Expected exactly 2 input datatypes, got %s", inputDataTypes);
        Preconditions.checkState(inputDataTypes.get(1).isIntType(), "Datatype for input 1 (Segment IDs) must be an integer type, got %s", inputDataTypes.get(1));
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
