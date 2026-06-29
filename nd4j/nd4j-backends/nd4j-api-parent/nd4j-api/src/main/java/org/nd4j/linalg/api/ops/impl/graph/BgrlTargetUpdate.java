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
 *  * specific language governing permissions and limitations under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.graph;

import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * BGRL (Bootstrap Your Own Latent for Graphs) target-network update utility.
 *
 * <p>BGRL (Thakoor et al. 2022) maintains two encoders: an <em>online</em> encoder trained
 * by gradient descent and a <em>target</em> encoder updated via exponential moving average
 * (EMA) of the online encoder's parameters. This class provides the EMA update rule for
 * a single parameter array.
 *
 * <p>Update rule:
 * <pre>
 *   target = momentum * target + (1 - momentum) * online
 * </pre>
 *
 * <p>Typical usage: after each gradient step on the online encoder, call
 * {@link #emaUpdate(INDArray, INDArray, double)} for every matching pair of
 * (target, online) parameter arrays, with momentum usually in [0.99, 0.999].
 */
public class BgrlTargetUpdate {

    private BgrlTargetUpdate() {}

    /**
     * Compute the EMA-updated target: {@code momentum * target + (1 - momentum) * online}.
     *
     * <p>The result is a <em>new</em> array; neither input is mutated. Both arrays must have
     * the same shape. {@code momentum} must be in (0, 1).
     *
     * @param target   current target-encoder parameter array (any numeric dtype)
     * @param online   current online-encoder parameter array (same shape as target)
     * @param momentum EMA decay factor, typically in [0.99, 0.999]
     * @return updated target array = momentum * target + (1 - momentum) * online
     */
    public static INDArray emaUpdate(INDArray target, INDArray online, double momentum) {
        Preconditions.checkArgument(momentum > 0.0 && momentum < 1.0,
                "momentum must be in (0, 1); got %s", momentum);
        Preconditions.checkArgument(target.equalShapes(online),
                "target and online must have the same shape; got %s vs %s",
                target.shapeInfoToString(), online.shapeInfoToString());
        return target.mul(momentum).addi(online.mul(1.0 - momentum));
    }

    /**
     * In-place variant: updates {@code target} by
     * {@code target = momentum * target + (1 - momentum) * online}.
     *
     * <p>Mutates {@code target} in place (does not allocate a new array for the final result).
     * Both arrays must have the same shape. {@code momentum} must be in (0, 1).
     *
     * @param target   current target-encoder parameter array (mutated in place)
     * @param online   current online-encoder parameter array (same shape as target)
     * @param momentum EMA decay factor, typically in [0.99, 0.999]
     */
    public static void emaUpdateInPlace(INDArray target, INDArray online, double momentum) {
        Preconditions.checkArgument(momentum > 0.0 && momentum < 1.0,
                "momentum must be in (0, 1); got %s", momentum);
        Preconditions.checkArgument(target.equalShapes(online),
                "target and online must have the same shape; got %s vs %s",
                target.shapeInfoToString(), online.shapeInfoToString());
        target.muli(momentum).addi(online.mul(1.0 - momentum));
    }
}
