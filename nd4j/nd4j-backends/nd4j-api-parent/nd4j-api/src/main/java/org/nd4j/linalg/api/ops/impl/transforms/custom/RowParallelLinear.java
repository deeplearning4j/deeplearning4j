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

import java.util.Collections;
import java.util.List;

/**
 * Row-parallel linear layer for tensor parallelism.
 * <p>
 * Splits the weight matrix along rows across tensor-parallel ranks.
 * Each rank computes a partial result, optionally reducing across ranks:
 * <pre>
 *   partial = inputShard @ weightShard
 *   output = allReduce(partial) + bias if reduceOutput else partial
 * </pre>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: inputShard [B, I/tp] - row-sharded input for this rank</li>
 *   <li>1: weightShard [I/tp, O] - row-sharded weight for this rank</li>
 *   <li>2: bias [O] (optional) - full bias (only added after reduce)</li>
 * </ul>
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: tpSize (default 1) - tensor parallel world size</li>
 *   <li>1: tpRank (default 0) - this rank's index</li>
 *   <li>2: reduceOutput (0=no, 1=yes, default 1) - whether to all-reduce output</li>
 * </ul>
 * <p>
 * Output: [B, O]
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class RowParallelLinear extends DynamicCustomOp {

    @Getter private int tpSize = 1;
    @Getter private int tpRank = 0;
    @Getter private int reduceOutput = 1;

    /**
     * SameDiff constructor with required inputs.
     */
    public RowParallelLinear(SameDiff sameDiff, SDVariable inputShard, SDVariable weightShard) {
        super(null, sameDiff, new SDVariable[]{inputShard, weightShard}, false);
        addIArgument((long) tpSize, (long) tpRank, (long) reduceOutput);
    }

    /**
     * SameDiff constructor with bias.
     */
    public RowParallelLinear(SameDiff sameDiff, SDVariable inputShard, SDVariable weightShard,
                             SDVariable bias) {
        super(null, sameDiff, new SDVariable[]{inputShard, weightShard, bias}, false);
        addIArgument((long) tpSize, (long) tpRank, (long) reduceOutput);
    }

    /**
     * SameDiff constructor with boolean reduceOutput.
     */
    public RowParallelLinear(SameDiff sameDiff, SDVariable inputShard, SDVariable weightShard,
                             SDVariable bias, int tpSize, int tpRank, boolean reduceOutput) {
        this(sameDiff, inputShard, weightShard, bias, tpSize, tpRank, reduceOutput ? 1 : 0);
    }

    /**
     * Full SameDiff constructor with all options.
     */
    public RowParallelLinear(SameDiff sameDiff, SDVariable inputShard, SDVariable weightShard,
                             SDVariable bias, int tpSize, int tpRank, int reduceOutput) {
        super(null, sameDiff, bias != null ?
                new SDVariable[]{inputShard, weightShard, bias} :
                new SDVariable[]{inputShard, weightShard}, false);
        this.tpSize = tpSize;
        this.tpRank = tpRank;
        this.reduceOutput = reduceOutput;
        addIArgument((long) tpSize, (long) tpRank, (long) reduceOutput);
    }

    /**
     * INDArray constructor (no output pre-allocation).
     */
    public RowParallelLinear(INDArray inputShard, INDArray weightShard, INDArray bias,
                             int tpSize, int tpRank, boolean reduceOutput) {
        this(inputShard, weightShard, bias, null, tpSize, tpRank, reduceOutput ? 1 : 0);
    }

    /**
     * INDArray constructor.
     */
    public RowParallelLinear(INDArray inputShard, INDArray weightShard, INDArray bias,
                             INDArray output, int tpSize, int tpRank, int reduceOutput) {
        super(null, bias != null ?
                new INDArray[]{inputShard, weightShard, bias} :
                new INDArray[]{inputShard, weightShard},
                output != null ? new INDArray[]{output} : null);
        this.tpSize = tpSize;
        this.tpRank = tpRank;
        this.reduceOutput = reduceOutput;
        addIArgument((long) tpSize, (long) tpRank, (long) reduceOutput);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.tpSize = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.tpRank = iArguments.get(1).intValue();
        if (iArguments.size() > 2) this.reduceOutput = iArguments.get(2).intValue();
    }

    @Override
    public String opName() {
        return "row_parallel_linear";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public int getNumOutputs() {
        return 1;
    }
}
