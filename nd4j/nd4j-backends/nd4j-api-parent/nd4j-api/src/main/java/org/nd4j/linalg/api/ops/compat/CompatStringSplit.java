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

package org.nd4j.linalg.api.ops.compat;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

/**
 * This is a wrapper for StringSplit op that impelements corresponding TF operation
 *
 * @author raver119@gmail.com
 */
public class CompatStringSplit extends DynamicCustomOp {

    public CompatStringSplit() {
        //
    }

    public CompatStringSplit(INDArray strings, INDArray delimiter) {
        Preconditions.checkArgument(strings.isS() && delimiter.isS(), "Input arrays must have one of UTF types");
        inputArguments.add(strings);
        inputArguments.add(delimiter);
    }

    public CompatStringSplit(INDArray strings, INDArray delimiter, INDArray indices, INDArray values) {
        this(strings, delimiter);

        outputArguments.add(indices);
        outputArguments.add(values);
    }

    @Override
    public String opName() {
        return "compat_string_split";
    }
}
