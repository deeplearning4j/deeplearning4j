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

package org.eclipse.deeplearning4j.nd4j.linalg.api.ops.impl.transforms.clip;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.shade.guava.primitives.Longs;

import java.util.Collections;
import java.util.List;

/**
 * The gradient of {@link org.nd4j.linalg.api.ops.impl.transforms.clip.ClipByAvgNorm}: from the op's input and the
 * gradient at its output, the gradient at its input.
 */
public class ClipByAvgNormBp extends DynamicCustomOp {

    private double clipValue;

    public ClipByAvgNormBp() {
    }

    public ClipByAvgNormBp(SameDiff sameDiff, SDVariable x, SDVariable eps, double clipValue, long... dimensions) {
        super(null, sameDiff, new SDVariable[]{x, eps});
        this.clipValue = clipValue;
        this.dimensions = dimensions;
        addIArgument(dimensions);
        addTArgument(clipValue);
    }

    @Override
    public String opName() {
        return "clipbyavgnorm_bp";
    }

    /** The integer arguments are the dimensions (none: the whole array), the floating point one the clip value. */
    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        this.dimensions = Longs.toArray(iArguments);
        if (!tArguments.isEmpty()) {
            this.clipValue = tArguments.get(0);
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 2,
                "Expected exactly 2 input datatypes for %s, got %s", getClass(), inputDataTypes);
        return Collections.singletonList(inputDataTypes.get(0));
    }
}
