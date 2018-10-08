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

package org.nd4j.linalg.api.ops.impl.shape;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Returns the size of the input along given dimension as a rank 0 array
 *
 * @author raver119@protonmail.com
 */
public class SizeAt extends DynamicCustomOp {

    public SizeAt() {}

    public SizeAt(INDArray input, int dimension) {
        this(input, null, dimension);
    }

    public SizeAt(INDArray input, INDArray output, int dimension) {
        super(null,input, output, new ArrayList<Double>(), new int[]{dimension});
    }

    public SizeAt(SameDiff sameDiff, SDVariable input, int dimension) {
        super(null, sameDiff, new SDVariable[] {input}, false);

        iArguments.add(Long.valueOf(dimension));
    }

    @Override
    public String opName() {
        return "size_at";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }
}
