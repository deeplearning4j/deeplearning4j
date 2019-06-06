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

package org.nd4j.linalg.api.blas.params;

import lombok.Data;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.BaseSparseNDArrayCOO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Audrey Loeffel
 */
@Data
public class SparseCOOGemvParameters {

    private int m, nnz;
    DataBuffer val, rowInd, colInd;
    private INDArray a, x, y;
    private char aOrdering = 'N';

    public SparseCOOGemvParameters(INDArray a, INDArray x, INDArray y) {
        this.a = a;
        this.x = x;
        this.y = y;

        if (a.isMatrix() && a.getFormat() == SparseFormat.COO) {
            BaseSparseNDArrayCOO coo = (BaseSparseNDArrayCOO) a;
            val = coo.getIncludedValues();
            nnz = coo.nnz();

            // FIXME: int cast
            m = (int) coo.rows();
            setIndexes(coo, false);
        }
    }

    private void setIndexes(BaseSparseNDArrayCOO coo, boolean oneBased) {
        int incr = oneBased ? 1 : 0;
        int[] idx = coo.getIncludedIndices().asInt();
        int[] rows = new int[nnz];
        int[] cols = new int[nnz];
        for (int i = 0; i < nnz; i++) {
            rows[i] = idx[i * 2] + incr;
            cols[i] = idx[(i * 2) + 1] + incr;
        }
        rowInd = Nd4j.createBuffer(rows);
        colInd = Nd4j.createBuffer(cols);
    }

}
