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

package org.nd4j.linalg.api.ops.exception;

import org.nd4j.linalg.api.blas.BlasBufferUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;

import java.io.Serializable;
import java.util.Arrays;

/**
 * @author Adam Gibson
 */
public class BlasOpErrorMessage implements Serializable {
    private Op op;

    public BlasOpErrorMessage(Op op) {
        this.op = op;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder().append("Op " + op.opName() + " of length " + op.n())
                        .append(" will fail with x of " + shapeInfo(op.x()));
        if (op.y() != null) {
            sb.append(" y of " + shapeInfo(op.y()));
        }

        sb.append(" and z of " + op.z());
        return sb.toString();
    }

    private String shapeInfo(INDArray arr) {
        return Arrays.toString(arr.shape()) + " and stride " + Arrays.toString(arr.stride()) + " and offset "
                        + arr.offset() + " and blas stride of " + BlasBufferUtil.getBlasStride(arr);
    }


}
