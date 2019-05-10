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

package org.nd4j.linalg.jcublas;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArrayCOO;

/**
 * @author Audrey Loeffel
 */
public class JCusparseNDArrayCOO extends BaseSparseNDArrayCOO {
    public JCusparseNDArrayCOO(DataBuffer values, DataBuffer indices, long[] shape) {
        super(values, indices, shape);
    }

    public JCusparseNDArrayCOO(double[] values, long[][] indices, long[] shape) {
        super(values, indices, shape);
    }

    public JCusparseNDArrayCOO(float[] values, long[][] indices, long[] shape) {
        super(values, indices, shape);
    }

    public JCusparseNDArrayCOO(double[] values, int[][] indices, long[] shape) {
        super(values, indices, shape);
    }

    public JCusparseNDArrayCOO(float[] values, int[][] indices, long[] shape) {
        super(values, indices, shape);
    }

    public JCusparseNDArrayCOO(DataBuffer values, DataBuffer indices, DataBuffer sparseInformation, long[] shape) {
        super(values, indices, sparseInformation, shape);
    }

    public JCusparseNDArrayCOO(DataBuffer values, DataBuffer indices, long[] sparseOffsets, int[] flags, int[] hiddenDimensions, int underlyingRank, long[] shape) {
        super(values, indices, sparseOffsets, flags, hiddenDimensions, underlyingRank, shape);
    }
}
