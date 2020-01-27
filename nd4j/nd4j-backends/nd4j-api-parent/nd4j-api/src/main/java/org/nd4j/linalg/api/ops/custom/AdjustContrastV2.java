/* ******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
package org.nd4j.linalg.api.ops.custom;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

public class AdjustContrastV2 extends BaseAdjustContrast {

    public AdjustContrastV2() {super();}

    public AdjustContrastV2(@NonNull INDArray in, double factor, INDArray out) {
        super(in, factor, out);
    }

    public AdjustContrastV2(@NonNull SameDiff sameDiff, @NonNull SDVariable in, @NonNull SDVariable factor) {
        super( sameDiff,new SDVariable[]{in,factor});
    }

    @Override
    public String opName() {
        return "adjust_contrast_v2";
    }

    @Override
    public String tensorflowName() {
        return "AdjustContrastv2";
    }
}