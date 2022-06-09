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

package org.nd4j.autodiff.samediff.internal.memory;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.autodiff.samediff.config.*;
import java.util.*;

public class WrapSDValue {

    public List<Long> idList;
    public SDValue value;

    public WrapSDValue(SDValue value) {
        this.value = value;
        this.idList = WrapSDValue.getIds(value);
    }

    public static List<Long> getIds(SDValue value) {
        List<Long> idList = new ArrayList<>();
        switch (value.getSdValueType()) {
            case LIST: {
                List<INDArray> listValue = value.getListValue();
                for (INDArray arr : listValue) {
                    if (arr != null)
                        idList.add(arr.getId());
                }
            }
            break;
            case TENSOR: {
                INDArray arr = value.getTensorValue();
                if (arr != null && arr.data() != null)
                    idList.add(arr.getId());
            }
            break;
        }
        return idList;
    }

    @Override
    public boolean equals(Object o) {
        WrapSDValue wrapped = (WrapSDValue) o;
        List<Long> listx = wrapped.idList;
        return listx.equals(this.idList);
    }

    @Override
    public int hashCode() {
        return this.idList.hashCode();
    }

}
