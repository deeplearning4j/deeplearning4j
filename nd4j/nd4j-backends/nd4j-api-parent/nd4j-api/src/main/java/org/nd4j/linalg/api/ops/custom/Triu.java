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
package org.nd4j.linalg.api.ops.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

@NoArgsConstructor
public class Triu extends DynamicCustomOp {

    private int diag = 0;

    public Triu(SameDiff sameDiff, SDVariable in, int diag) {
        super(sameDiff, new SDVariable[]{in});
        addIArgument(diag);
        this.diag=diag;
    }

    public Triu(SameDiff sameDiff, SDVariable in) {
        super(sameDiff, new SDVariable[]{in});
    }



    public Triu(INDArray input, int diag) {
        super(new INDArray[]{input}, null);
        addIArgument(diag);
        this.diag=diag;

    }


    @Override
    public String opName() {
        return "triu";
    }


    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        return Collections.singletonList(arg(0).dataType());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {

        return new TriuBp(sameDiff, arg(0), f1.get(0), diag).outputs();
    }
}
