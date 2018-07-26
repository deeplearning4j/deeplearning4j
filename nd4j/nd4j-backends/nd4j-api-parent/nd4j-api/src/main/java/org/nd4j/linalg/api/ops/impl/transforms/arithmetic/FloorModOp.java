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

package org.nd4j.linalg.api.ops.impl.transforms.arithmetic;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.List;

/**
 * Floor mod
 *
 * @author raver119@gmail.com
 */
public class FloorModOp extends DynamicCustomOp {
    public FloorModOp() {}

    public FloorModOp(SameDiff sameDiff, SDVariable x, SDVariable y) {
        super(null, sameDiff, new SDVariable[]{x, y});
    }

    @Override
    public String opName() {
        return "floormod";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return f().floorModBp(larg(), rarg(), f1.get(0));
    }
}
