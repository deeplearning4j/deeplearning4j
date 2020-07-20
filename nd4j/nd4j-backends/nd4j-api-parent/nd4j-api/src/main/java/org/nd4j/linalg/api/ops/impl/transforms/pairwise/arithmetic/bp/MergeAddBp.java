/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.bp;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.List;

@NoArgsConstructor
public class MergeAddBp extends DynamicCustomOp {

    public MergeAddBp(SameDiff sameDiff, @NonNull SDVariable[] inputs, @NonNull SDVariable gradO) {
        super("mergeadd_bp", sameDiff, ArrayUtils.add(inputs, gradO));
    }

    @Override
    public String opName() {
        return "mergeadd_bp";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        ArrayList<DataType> list = new ArrayList<DataType>();
        for (int i=0; i< args().length-1;i++){list.add(inputDataTypes.get(0));}
        return list;

    }

    @Override
    public int getNumOutputs(){
        return args().length-1;
    }
}
