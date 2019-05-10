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

package org.nd4j.linalg.api.ndarray;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author Audrey Loeffel
 */
public interface ISparseNDArray extends INDArray {
    /*
    * TODO
    * Will contain methods such as toDense, toCSRFormat,...
    *
    * */

    /**
     * Return a array of non-major pointers
     * i.e. return the column indexes in case of row-major ndarray
     * @return a DataBuffer of indexes
     * */
    DataBuffer getVectorCoordinates();

    /**
     * Return a dense representation of the sparse ndarray
     * */
    INDArray toDense();

    /**
     * Return the number of non-null element
     * @return nnz
     * */
    int nnz();

    /**
     * Return the sparse format (i.e COO, CSR, ...)
     * @return format
     * @see SparseFormat
     * */
    SparseFormat getFormat();
}
