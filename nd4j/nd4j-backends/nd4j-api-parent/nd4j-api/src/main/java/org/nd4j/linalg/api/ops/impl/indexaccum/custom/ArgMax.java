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

package org.nd4j.linalg.api.ops.impl.indexaccum.custom;

import lombok.Data;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.reduce.custom.BaseDynamicCustomIndexReduction;

import java.util.List;

@Data
public class ArgMax extends BaseDynamicCustomIndexReduction {
    public ArgMax(SameDiff sameDiff, SDVariable[] args, boolean keepDims) {
        super(sameDiff, args, keepDims);
    }

    public ArgMax(SameDiff sameDiff, SDVariable[] args, boolean keepDims, long[] dimensions) {
        super(sameDiff, args, keepDims, dimensions);
    }

    public ArgMax(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex) {
        super(sameDiff, args, keepDims, isComplex);
    }

    public ArgMax(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, long[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, dimensions);
    }


    public ArgMax(INDArray[] inputs) {
        super(inputs, null);
    }


    public ArgMax(INDArray[] inputs, INDArray[] outputs) {
        super(inputs, outputs);
    }

    public ArgMax(INDArray[] inputs, INDArray[] outputs, boolean keepDims) {
        super(inputs, outputs, keepDims);
    }

    public ArgMax(INDArray[] inputs, INDArray[] outputs, boolean keepDims, long... dimensions) {
        super(inputs, outputs, keepDims, dimensions);
    }

    public ArgMax(INDArray[] inputs, boolean keepDims, long[] dimensions) {
        super(inputs, keepDims, dimensions);
    }

    public ArgMax(boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public ArgMax(SameDiff sameDiff, SDVariable arg, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(sameDiff, arg, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public ArgMax(SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public ArgMax(String opName, SameDiff sameDiff, SDVariable[] args, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, sameDiff, args, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public ArgMax(String opName, INDArray input, INDArray output, List<Double> tArguments, long[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, input, output, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public ArgMax(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, long[] iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public ArgMax(String opName, INDArray[] inputs, INDArray[] outputs, List<Double> tArguments, List<Long> iArguments, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, inputs, outputs, tArguments, iArguments, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public ArgMax(INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public ArgMax(String opName, INDArray[] inputs, INDArray[] outputs, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, inputs, outputs, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public ArgMax(String opName, SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public ArgMax(SameDiff sameDiff, SDVariable[] args, boolean inPlace, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(sameDiff, args, inPlace, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public ArgMax(String opName, boolean keepDims, boolean isComplex, boolean isEmptyReduce, long[] dimensions) {
        super(opName, keepDims, isComplex, isEmptyReduce, dimensions);
    }

    public ArgMax(INDArray[] input, INDArray[] output, boolean keepDims, boolean isComplex, long[] dimensions) {
        super(input, output, keepDims, isComplex, dimensions);
    }

    public ArgMax() {
    }

    public ArgMax(SameDiff sd, SDVariable in, boolean keepDims, long[] dimensions) {
        this(sd,new SDVariable[]{in},keepDims,dimensions);
    }

    public ArgMax(INDArray in, boolean keepDims, long[] dimensions) {
        this(new INDArray[]{in},keepDims,dimensions);
    }

    public ArgMax(INDArray arr) {
        this(new INDArray[]{arr},false,null);
    }

    @Override
    public String opName() {
        return "argmax";
    }

    @Override
    public String tensorflowName() {
        return "ArgMax";
    }



}
