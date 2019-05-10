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

package org.nd4j.linalg.jcublas.blas;

import org.nd4j.linalg.api.blas.impl.SparseBaseLevel2;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Audrey Loeffel
 */
public class JcusparseLevel2 extends SparseBaseLevel2 {


    @Override
    protected void scoomv(char transA, int M, DataBuffer values, DataBuffer rowInd, DataBuffer colInd, int nnz, INDArray x, INDArray y) {

    }

    @Override
    protected void dcoomv(char transA, int M, DataBuffer values, DataBuffer rowInd, DataBuffer colInd, int nnz, INDArray x, INDArray y) {

    }
}
