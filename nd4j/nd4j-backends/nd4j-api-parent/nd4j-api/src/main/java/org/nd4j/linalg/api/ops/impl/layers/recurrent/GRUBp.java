/*
 *  ******************************************************************************
 *  *
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
package org.nd4j.linalg.api.ops.impl.layers.recurrent;


import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.ArrayList;
import java.util.List;

@NoArgsConstructor
public class GRUBp extends DynamicCustomOp {


    public GRUBp(@NonNull SameDiff sameDiff, @NonNull SDVariable x, @NonNull SDVariable hI, @NonNull SDVariable Wx, @NonNull SDVariable Wh, @NonNull SDVariable biases, @NonNull SDVariable dLdh) {
        super(null, sameDiff, new SDVariable[]{x, hI, Wx, Wh, biases, dLdh});

    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        DataType dt = inputDataTypes.get(1);
        List<DataType> list = new ArrayList<DataType>();
        list.add(dt);
        list.add(dt);
        list.add(dt);
        list.add(dt);
        list.add(dt);
        return list;
    }

    @Override
    public String opName() {
        return "gru_bp";
    }
}
