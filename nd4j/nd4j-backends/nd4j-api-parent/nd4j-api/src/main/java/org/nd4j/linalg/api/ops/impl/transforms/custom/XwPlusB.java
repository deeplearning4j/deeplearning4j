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

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.*;


@NoArgsConstructor
public class XwPlusB extends DynamicCustomOp {


    private boolean aTranspose,bTranspose,cTranspose;


    public XwPlusB(SameDiff sameDiff, SDVariable input, SDVariable weights, SDVariable bias) {
        super(null, sameDiff, new SDVariable[] {input, weights, bias}, false);
    }

    public XwPlusB(INDArray input, INDArray weights, INDArray bias) {
        super(new INDArray[] {input, weights, bias}, null);
    }

    public XwPlusB(INDArray[] inputs, INDArray output){
        super(inputs, wrapOrNull(output));
    }

    public XwPlusB(SameDiff sd, SDVariable input, SDVariable weights, SDVariable bias, boolean transposeA, boolean transposeB, boolean transposeC) {
        super(null,sd,new SDVariable[]{input,weights,bias});
        addIArgument(transposeA ? 1 : 0, transposeB ? 1 : 0,transposeC ? 1 : 0);
        this.aTranspose = transposeA;
        this.bTranspose = transposeB;
        this.cTranspose = transposeC;
    }

    public XwPlusB(INDArray input, INDArray weights, INDArray bias, boolean transposeA, boolean transposeB, boolean transposeC) {
        super(null,new INDArray[]{input,weights,bias},null);
        addIArgument(transposeA ? 1 : 0,transposeB ? 1 : 0,transposeC ? 1 : 0);
    }

    @Override
    public String opName() {
        return "xw_plus_b";
    }


    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow name found for shape " + opName());
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }

    @Override
    public void configureFromArguments() {
        if(!iArguments.isEmpty()) {
            if(iArguments.size() == 1) {
                this.aTranspose = iArguments.get(0) > 0;
            }

            if(iArguments.size() > 1) {
                this.bTranspose = iArguments.get(1) > 0;
            }

            if(iArguments.size() > 2) {
                this.cTranspose = iArguments.get(2) > 0;
            }


        }
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradient) {
        return Arrays.asList(new XwPlusBBp(
                sameDiff,
                arg(0),
                arg(1),
                arg(2),
                gradient.get(0),
                aTranspose,bTranspose).outputVariables());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 3, "Expected exactly 3 input datatypes, got %s", dataTypes);
        DataType first = dataTypes.get(0);
        return Collections.singletonList(first);
    }

}
