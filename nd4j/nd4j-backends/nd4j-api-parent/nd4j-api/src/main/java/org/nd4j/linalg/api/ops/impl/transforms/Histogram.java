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

package org.nd4j.linalg.api.ops.impl.transforms;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

/**
 * Histogram op wrapper
 *
 * @author raver119@gmail.com
 */
public class Histogram extends DynamicCustomOp {
    private long numBins;

    public Histogram() {

    }

    public Histogram(INDArray input, INDArray output) {
        Preconditions.checkArgument(output.isZ(), "Histogram op output should have integer data type");

        numBins = output.length();
        inputArguments.add(input);
        outputArguments.add(output);
        iArguments.add(numBins);
    }

    public Histogram(INDArray input, long numBins) {
        this.numBins = numBins;
        inputArguments.add(input);
        iArguments.add(numBins);
    }

    @Override
    public String opName() {
        return "histogram";
    }
}
