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

package org.nd4j.linalg.api.ops.impl.grid;

import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.GridOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.grid.GridDescriptor;
import org.nd4j.linalg.api.ops.grid.GridPointers;
import org.nd4j.linalg.api.ops.grid.OpDescriptor;

import java.util.ArrayList;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public abstract class BaseGridOp extends BaseOp implements GridOp {
    protected List<OpDescriptor> queuedOps = new ArrayList<>();
    protected List<GridPointers> grid = new ArrayList<>();

    public BaseGridOp() {

    }

    public BaseGridOp(INDArray x, INDArray y) {
        // no-op
    }

    protected BaseGridOp(Op... ops) {
        grid = new ArrayList<>(ops.length);
        for (Op op : ops) {
            queuedOps.add(new OpDescriptor(op, null));
            grid.add(null);
        }
    }

    protected BaseGridOp(OpDescriptor... descriptors) {
        for (OpDescriptor op : descriptors) {
            queuedOps.add(op);
            grid.add(null);
        }
    }

    protected BaseGridOp(GridPointers... pointers) {
        for (GridPointers ptr : pointers) {
            grid.add(ptr);
        }
    }

    protected BaseGridOp(List<Op> ops) {
        this(ops.toArray(new Op[0]));
    }


    @Override
    public GridDescriptor getGridDescriptor() {
        GridDescriptor descriptor = new GridDescriptor();
        descriptor.setGridDepth(grid.size());
        descriptor.setGridPointers(grid);
        return descriptor;
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " + opName());
    }
}
