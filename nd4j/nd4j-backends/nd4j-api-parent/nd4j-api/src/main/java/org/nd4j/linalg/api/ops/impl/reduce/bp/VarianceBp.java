/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.reduce.bp;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;


@NoArgsConstructor
public class VarianceBp extends BaseReductionBp {

    private boolean biasCorrected;

    public VarianceBp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean biasCorrected, boolean keepDims, int... dimensions) {
        super(sameDiff, origInput, gradAtOutput, keepDims, dimensions);
        this.biasCorrected = biasCorrected;
        addArgs();

    }

    public VarianceBp(SameDiff sameDiff, SDVariable origInput1, SDVariable origInput2, SDVariable gradAtOutput, boolean keepDims, boolean biasCorrected, int... dimensions) {
        super(sameDiff, origInput1, origInput2, gradAtOutput, keepDims, dimensions);
        this.biasCorrected = biasCorrected;
        addArgs();

    }

    public VarianceBp(INDArray origInput, INDArray gradAtOutput, INDArray output, boolean biasCorrected, boolean keepDims, int... dimensions){
        super(origInput, gradAtOutput, output, keepDims, dimensions);
        this.biasCorrected = biasCorrected;
        addArgs();

    }

    public VarianceBp(INDArray origInput1, INDArray origInput2, INDArray gradAtOutput, INDArray output, boolean keepDims, boolean biasCorrected, int... dimensions) {
        super(origInput1, origInput2, gradAtOutput, output, keepDims, dimensions);
        this.biasCorrected = biasCorrected;
        addArgs();

    }

    public VarianceBp(INDArray origInput1, INDArray origInput2, INDArray gradAtOutput, INDArray output1, INDArray output2, boolean keepDims, boolean biasCorrected, int... dimensions) {
        super(origInput1, origInput2, gradAtOutput, output1, output2, keepDims, dimensions);
        this.biasCorrected = biasCorrected;
        addArgs();

    }

    public VarianceBp(INDArray origInput, INDArray gradAtOutput, INDArray output, boolean keepDims, INDArray dimensions, boolean biasCorrected) {
        super(origInput, gradAtOutput, output, keepDims, dimensions);
        this.biasCorrected = biasCorrected;
        addArgs();

    }

    public VarianceBp(SameDiff sameDiff, SDVariable origInput, SDVariable gradAtOutput, boolean keepDims, SDVariable dimensions, boolean biasCorrected) {
        super(sameDiff, origInput, gradAtOutput, keepDims, dimensions);
        this.biasCorrected = biasCorrected;
        addArgs();
    }

    public VarianceBp(SameDiff sameDiff, SDVariable arg, SDVariable dLdVar, boolean keepDims, boolean biasCorrected, SDVariable dimensions) {
        super(sameDiff,arg,dLdVar,keepDims,dimensions);
        addBArgument(biasCorrected);
    }

    @Override
    protected void addArgs() {
        super.addArgs();
        addBArgument(biasCorrected);
    }

    public VarianceBp(boolean biasCorrected) {
        this.biasCorrected = biasCorrected;
    }

    @Override
    public String opName() {
        return "reduce_variance_bp";
    }
}
