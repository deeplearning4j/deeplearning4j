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
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * Unsorted segment mean operation
 *
 * @author Alex Black
 */
public class UnsortedSegmentMean extends DynamicCustomOp {

    private int numSegments;

    public UnsortedSegmentMean(SameDiff sameDiff, SDVariable data, SDVariable segmentIds, int numSegments) {
        super(null, sameDiff,  new SDVariable[] {data, segmentIds}, false);
        this.numSegments = numSegments;
        addIArgument(numSegments);
    }

    public UnsortedSegmentMean(){ }

    @Override
    public String opName(){
        return "unsorted_segment_mean";
    }

    @Override
    public String tensorflowName() {
        return "UnsortedSegmentMean";
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
        return Arrays.asList(f().unsortedSegmentMeanBp(arg(0), arg(1), gradients.get(0), numSegments));
    }

}
