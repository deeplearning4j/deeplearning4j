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

package org.nd4j.linalg.api.ops.impl.scatter;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * @author raver119@protonmail.com
 * @author Alex Black
 */

public class ScatterMin extends DynamicCustomOp {

    public ScatterMin(SameDiff sameDiff, SDVariable ref, SDVariable indices, SDVariable updates) {
        super(null, sameDiff, new SDVariable[]{ref, indices, updates}, false);
    }

    public ScatterMin() {}

    @Override
    public String opName() {
        return "scatter_min";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "ScatterMin";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradOut) {
        //3 args: ref, indices, updates
        //For non-modified indices, input gradient (reference) is same as output gradient
        //For modified indices, dL/dref = dL/dOut if(ref[index[i],j] == min) or 0 otherwise
        //And for updates, dL/du = dL/dOut if(update[i,j]==min) or 0 otherwise

        List<SDVariable> ret = new ArrayList<>(3);
        SDVariable notModified = arg(0).eq(outputVariable());   //0 if modified, 1 otherwise
        SDVariable refGrad = gradOut.get(0).mul(notModified);

        SDVariable gatherOut = f().gather(outputVariable(), arg(1), 0);
        SDVariable gatherGrad = f().gather(gradOut.get(0), arg(1), 0);
        SDVariable outIsUpdate = gatherOut.eq(arg(2));
        SDVariable updateGrad = gatherGrad.mul(outIsUpdate);

        return Arrays.asList(refGrad, f().zerosLike(arg(1)), updateGrad);
    }

}
