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

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Mamba-2 Structured State Space Model (SSM) op.
 *
 * Implements the discrete SSM recurrence from Mamba-2:
 *   state_out = A * state_in + B * x
 *   y = C * state_out + D * x  (D is optional)
 *
 * Produces two outputs:
 *   0: y         - the output sequence
 *   1: state_out - the updated SSM state
 *
 * Inputs:
 *   0: x       - input tensor
 *   1: A       - state transition matrix
 *   2: B       - input projection matrix
 *   3: C       - output projection matrix
 *   4: dt      - discretization timestep
 *   5: D       - (optional) skip connection weights
 *   6: stateIn - (optional) initial state
 *
 * iArgs:
 *   0: numHeads  - number of SSM heads
 *   1: headDim   - dimension per head
 *   2: stateDim  - SSM state dimension
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class Mamba2SSM extends DynamicCustomOp {

    @Getter private int numHeads;
    @Getter private int headDim;
    @Getter private int stateDim;

    /**
     * Full constructor with all inputs including optional D and stateIn.
     */
    public Mamba2SSM(INDArray x, INDArray A, INDArray B, INDArray C, INDArray dt,
                     INDArray D, INDArray stateIn,
                     int numHeads, int headDim, int stateDim) {
        super(buildInputs(x, A, B, C, dt, D, stateIn), null);
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.stateDim = stateDim;
        addIArgument(numHeads, headDim, stateDim);
    }

    /**
     * Constructor with required inputs only (no D, no stateIn).
     */
    public Mamba2SSM(INDArray x, INDArray A, INDArray B, INDArray C, INDArray dt,
                     int numHeads, int headDim, int stateDim) {
        this(x, A, B, C, dt, null, null, numHeads, headDim, stateDim);
    }

    /**
     * Full SameDiff constructor with all inputs including optional D and stateIn.
     */
    public Mamba2SSM(SameDiff sd, SDVariable x, SDVariable A, SDVariable B, SDVariable C, SDVariable dt,
                     SDVariable D, SDVariable stateIn,
                     int numHeads, int headDim, int stateDim) {
        super(null, sd, buildSdInputs(x, A, B, C, dt, D, stateIn), false);
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.stateDim = stateDim;
        addIArgument(numHeads, headDim, stateDim);
    }

    /**
     * SameDiff constructor with required inputs only (no D, no stateIn).
     */
    public Mamba2SSM(SameDiff sd, SDVariable x, SDVariable A, SDVariable B, SDVariable C, SDVariable dt,
                     int numHeads, int headDim, int stateDim) {
        this(sd, x, A, B, C, dt, null, null, numHeads, headDim, stateDim);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.numHeads = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.headDim = iArguments.get(1).intValue();
        if (iArguments.size() > 2) this.stateDim = iArguments.get(2).intValue();
    }

    @Override
    public String opName() {
        return "mamba2_ssm";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        DataType dt = inputDataTypes.get(0);
        return Arrays.asList(dt, dt);
    }

    @Override
    public int getNumOutputs() {
        return 2;
    }

    private static INDArray[] buildInputs(INDArray x, INDArray A, INDArray B, INDArray C, INDArray dt,
                                          INDArray D, INDArray stateIn) {
        if (D != null && stateIn != null) {
            return new INDArray[]{x, A, B, C, dt, D, stateIn};
        } else if (D != null) {
            return new INDArray[]{x, A, B, C, dt, D};
        } else if (stateIn != null) {
            return new INDArray[]{x, A, B, C, dt, stateIn};
        } else {
            return new INDArray[]{x, A, B, C, dt};
        }
    }

    private static SDVariable[] buildSdInputs(SDVariable x, SDVariable A, SDVariable B, SDVariable C, SDVariable dt,
                                              SDVariable D, SDVariable stateIn) {
        if (D != null && stateIn != null) {
            return new SDVariable[]{x, A, B, C, dt, D, stateIn};
        } else if (D != null) {
            return new SDVariable[]{x, A, B, C, dt, D};
        } else if (stateIn != null) {
            return new SDVariable[]{x, A, B, C, dt, stateIn};
        } else {
            return new SDVariable[]{x, A, B, C, dt};
        }
    }
}
