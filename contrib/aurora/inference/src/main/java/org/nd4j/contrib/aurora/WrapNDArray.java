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

package org.nd4j.contrib.aurora;

import org.nd4j.linalg.api.ndarray.INDArray;

public class WrapNDArray {

    public INDArray arr;

    public WrapNDArray(INDArray arr) {
        this.arr = arr;
    }

    @Override
    public boolean equals(Object o) {
        WrapNDArray w = (WrapNDArray) o;
        return w.arr != null && (w.arr.getId() == this.arr.getId());
    }

    @Override
    public int hashCode() {
        return (int) (this.arr.getId() % Integer.MAX_VALUE);
    }

}
