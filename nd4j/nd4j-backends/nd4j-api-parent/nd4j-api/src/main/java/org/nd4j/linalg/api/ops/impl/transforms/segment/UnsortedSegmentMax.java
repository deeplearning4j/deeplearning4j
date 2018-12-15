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

package org.nd4j.linalg.api.ops.impl.transforms.segment;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * Unsorted segment max operation
 *
 * @author Alex Black
 */
public class UnsortedSegmentMax extends DynamicCustomOp {

    protected int numSegments;

    public UnsortedSegmentMax(SameDiff sameDiff, SDVariable data, SDVariable segmentIds, int numSegments) {
        super(null, sameDiff,  new SDVariable[] {data, segmentIds}, false);
        this.numSegments = numSegments;
        addIArgument(numSegments);
    }

    public UnsortedSegmentMax(){ }

    @Override
    public String opName(){
        return "unsorted_segment_max";
    }

    @Override
    public String tensorflowName() {
        return "UnsortedSegmentMax";
    }

    @Override
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        if(args().length == 3 && iArguments == null || iArguments.size() == 0){
            addIArgument(arg(2).getArr().getInt(0));
        }
        super.resolvePropertiesFromSameDiffBeforeExecution();
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradients){
        return Arrays.asList(f().unsortedSegmentMaxBp(arg(0), arg(1), gradients.get(0), numSegments));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 2, "Expected exactly 2 input data types, got %s", inputDataTypes);
        List<DataType> out = new ArrayList<>();
        for( int i=0; i<numSegments; i++ ){
            out.add(inputDataTypes.get(0));
        }
        return out;
    }

}
