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

package org.nd4j.linalg.api.ops.impl.meta;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;

import java.util.List;

/**
 * This is special case PredicateOp, with opB being only either Accumulation, Variance or Reduce3 op
 *
 * @author raver119@gmail.com
 */
public class ReduceMetaOp extends BaseMetaOp {

    public ReduceMetaOp() {
        super();
    }

    public ReduceMetaOp(ScalarOp opA, Accumulation opB, int... dimensions) {
        this(new OpDescriptor(opA), new OpDescriptor(opB, dimensions));
    }

    public ReduceMetaOp(INDArray x, INDArray y) {
        super(x, y);
    }

    public ReduceMetaOp(ScalarOp opA, Accumulation opB) {
        super(opA, opB);
    }

    public ReduceMetaOp(OpDescriptor opA, OpDescriptor opB) {
        super(opA, opB);
    }


    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName() {
        return "meta_reduce";
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }
}
