/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.linalg.lossfunctions;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;

/**
 * Created by Alex on 14/09/2016.
 */
public class LossUtil {

    /**
     *
     * @param to
     * @param mask
     * @return
     */
    public static boolean isPerOutputMasking(INDArray to, INDArray mask) {
        return !mask.isColumnVector() || Arrays.equals(to.shape(), mask.shape());
    }

    /**
     *
     * @param to
     * @param mask
     */
    public static void applyMask(INDArray to, INDArray mask) {
        //Two possibilities exist: it's *per example* masking, or it's *per output* masking
        //These cases have different mask shapes. Per example: column vector. Per output: same shape as score array
        if (mask.isColumnVectorOrScalar()) {
            to.muliColumnVector(mask);
        } else if (Arrays.equals(to.shape(), mask.shape())) {
            to.muli(mask);
        } else {
            throw new IllegalStateException("Invalid mask array: per-example masking should be a column vector, "
                            + "per output masking arrays should be the same shape as the labels array. Mask shape: "
                            + Arrays.toString(mask.shape()) + ", output shape: " + Arrays.toString(to.shape()));
        }
    }
}
