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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.MetaOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.grid.GridPointers;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;
import org.nd4j.linalg.api.ops.impl.grid.BaseGridOp;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseMetaOp extends BaseGridOp implements MetaOp {

    public BaseMetaOp() {

    }

    public BaseMetaOp(INDArray x, INDArray y) {
        super(x, y);
    }

    protected BaseMetaOp(Op opA, Op opB) {
        super(opA, opB);
    }

    @Override
    public OpDescriptor getFirstOpDescriptor() {
        return queuedOps.get(0);
    }

    @Override
    public OpDescriptor getSecondOpDescriptor() {
        return queuedOps.get(1);
    }

    protected BaseMetaOp(OpDescriptor opA, OpDescriptor opB) {
        super(opA, opB);
    }

    protected BaseMetaOp(GridPointers opA, GridPointers opB) {
        super(opA, opB);
    }

    public Op getFirstOp() {
        return getFirstOpDescriptor().getOp();
    }

    public Op getSecondOp() {
        return getSecondOpDescriptor().getOp();
    }

    @Override
    public void setFirstPointers(GridPointers pointers) {
        grid.set(0, pointers);
    }

    @Override
    public void setSecondPointers(GridPointers pointers) {
        grid.set(1, pointers);
    }
}
