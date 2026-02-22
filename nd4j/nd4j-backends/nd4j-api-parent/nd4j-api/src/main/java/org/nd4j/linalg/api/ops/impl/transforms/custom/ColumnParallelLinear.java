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
 * Column-parallel linear layer for tensor parallelism.
 * <p>
 * Splits the weight matrix along columns across tensor-parallel ranks.
 * Each rank computes a shard of the output, optionally gathering across ranks:
 * <pre>
 *   output_shard = input @ weightShard + biasShard
 *   output = allGather(output_shard) if gatherOutput else output_shard
 * </pre>
 * <p>
 * Inputs:
 * <ul>
 *   <li>0: input [B, I] - input activations</li>
 *   <li>1: weightShard [I, O/tp] - column-sharded weight for this rank</li>
 *   <li>2: biasShard [O/tp] (optional) - bias shard for this rank</li>
 * </ul>
 * <p>
 * Integer arguments:
 * <ul>
 *   <li>0: tpSize (default 1) - tensor parallel world size</li>
 *   <li>1: tpRank (default 0) - this rank's index</li>
 *   <li>2: gatherOutput (0=no, 1=yes, default 1) - whether to all-gather output</li>
 * </ul>
 * <p>
 * Output: [B, O] if gatherOutput=1, else [B, O/tp]
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@NoArgsConstructor
public class ColumnParallelLinear extends DynamicCustomOp {

    @Getter private int tpSize = 1;
    @Getter private int tpRank = 0;
    @Getter private int gatherOutput = 1;

    /**
     * SameDiff constructor with required inputs.
     */
    public ColumnParallelLinear(SameDiff sameDiff, SDVariable input, SDVariable weightShard) {
        super(null, sameDiff, new SDVariable[]{input, weightShard}, false);
        addIArgument((long) tpSize, (long) tpRank, (long) gatherOutput);
    }

    /**
     * SameDiff constructor with bias.
     */
    public ColumnParallelLinear(SameDiff sameDiff, SDVariable input, SDVariable weightShard,
                                SDVariable biasShard) {
        super(null, sameDiff, new SDVariable[]{input, weightShard, biasShard}, false);
        addIArgument((long) tpSize, (long) tpRank, (long) gatherOutput);
    }

    /**
     * SameDiff constructor with boolean gatherOutput.
     */
    public ColumnParallelLinear(SameDiff sameDiff, SDVariable input, SDVariable weightShard,
                                SDVariable biasShard, int tpSize, int tpRank, boolean gatherOutput) {
        this(sameDiff, input, weightShard, biasShard, tpSize, tpRank, gatherOutput ? 1 : 0);
    }

    /**
     * Full SameDiff constructor with all options.
     */
    public ColumnParallelLinear(SameDiff sameDiff, SDVariable input, SDVariable weightShard,
                                SDVariable biasShard, int tpSize, int tpRank, int gatherOutput) {
        super(null, sameDiff, biasShard != null ?
                new SDVariable[]{input, weightShard, biasShard} :
                new SDVariable[]{input, weightShard}, false);
        this.tpSize = tpSize;
        this.tpRank = tpRank;
        this.gatherOutput = gatherOutput;
        addIArgument((long) tpSize, (long) tpRank, (long) gatherOutput);
    }

    /**
     * INDArray constructor (no output pre-allocation).
     */
    public ColumnParallelLinear(INDArray input, INDArray weightShard, INDArray biasShard,
                                int tpSize, int tpRank, boolean gatherOutput) {
        this(input, weightShard, biasShard, null, tpSize, tpRank, gatherOutput ? 1 : 0);
    }

    /**
     * INDArray constructor.
     */
    public ColumnParallelLinear(INDArray input, INDArray weightShard, INDArray biasShard,
                                INDArray output, int tpSize, int tpRank, int gatherOutput) {
        super(null, biasShard != null ?
                new INDArray[]{input, weightShard, biasShard} :
                new INDArray[]{input, weightShard},
                output != null ? new INDArray[]{output} : null);
        this.tpSize = tpSize;
        this.tpRank = tpRank;
        this.gatherOutput = gatherOutput;
        addIArgument((long) tpSize, (long) tpRank, (long) gatherOutput);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        if (iArguments.size() > 0) this.tpSize = iArguments.get(0).intValue();
        if (iArguments.size() > 1) this.tpRank = iArguments.get(1).intValue();
        if (iArguments.size() > 2) this.gatherOutput = iArguments.get(2).intValue();
    }

    @Override
    public String opName() {
        return "column_parallel_linear";
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
