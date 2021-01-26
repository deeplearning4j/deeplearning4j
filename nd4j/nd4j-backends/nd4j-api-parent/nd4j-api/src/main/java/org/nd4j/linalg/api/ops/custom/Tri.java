/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
public class Tri extends DynamicCustomOp {

    private DataType dataType = DataType.FLOAT;

    public Tri(SameDiff sameDiff, int row, int column, int diag) {
        super(sameDiff, new SDVariable[]{});
        addIArgument(row,column,diag);
    }

    public Tri(SameDiff sameDiff, DataType dataType, int row, int column, int diag) {
        super(sameDiff, new SDVariable[]{});
        addIArgument(row,column,diag);
        addDArgument(dataType);
        this.dataType = dataType;


    }

    public Tri(int row, int column, int diag) {
        super(new INDArray[]{}, null);
        addIArgument(row,column,diag);

    }

    public Tri(DataType dataType, int row, int column, int diag) {
        super(new INDArray[]{}, null);
        addIArgument(row,column,diag);
        addDArgument(dataType);
        this.dataType = dataType;

    }

    @Override
    public String opName() {
        return "tri";
    }


    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {

        return Collections.singletonList(this.dataType);

    }
}
