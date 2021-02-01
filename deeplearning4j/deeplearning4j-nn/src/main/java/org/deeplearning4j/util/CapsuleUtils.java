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

package org.deeplearning4j.util;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Utilities for CapsNet Layers
 * @see org.deeplearning4j.nn.conf.layers.CapsuleLayer
 * @see org.deeplearning4j.nn.conf.layers.PrimaryCapsules
 * @see org.deeplearning4j.nn.conf.layers.CapsuleStrengthLayer
 *
 * @author Ryan Nett
 */
public class CapsuleUtils {

    /**
     *  Compute the squash operation used in CapsNet
     *  The formula is (||s||^2 / (1 + ||s||^2)) * (s / ||s||).
     *  Canceling one ||s|| gives ||s||*s/((1 + ||s||^2)
     *
     * @param SD The SameDiff environment
     * @param x The variable to squash
     * @return squash(x)
     */
    public static SDVariable squash(SameDiff SD, SDVariable x, int dim){
        SDVariable squaredNorm = SD.math.square(x).sum(true, dim);
        SDVariable scale = SD.math.sqrt(squaredNorm.plus(1e-5));
        return x.times(squaredNorm).div(squaredNorm.plus(1.0).times(scale));
    }

}
