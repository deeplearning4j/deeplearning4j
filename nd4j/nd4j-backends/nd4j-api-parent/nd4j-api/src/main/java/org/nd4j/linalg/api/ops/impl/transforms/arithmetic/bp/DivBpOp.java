/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 *
 */

package org.nd4j.linalg.api.ops.impl.transforms.arithmetic.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Division backprop operation. Supports 'undoing' of auto broadcast as applied in div op forward pass
 *
 * @author Alex Black
 */
public class DivBpOp extends BaseArithmeticBackpropOp {

    public DivBpOp() {}

    public DivBpOp(SameDiff sameDiff, SDVariable x, SDVariable y, SDVariable eps) {
        super(sameDiff, x,y,eps);
    }

    @Override
    public String opName() {
        return "divide_bp";
    }
}
