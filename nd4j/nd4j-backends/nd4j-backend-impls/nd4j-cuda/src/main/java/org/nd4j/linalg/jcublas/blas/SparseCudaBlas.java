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


import org.nd4j.linalg.api.blas.Blas;
import org.nd4j.nativeblas.SparseNd4jBlas;

/**
 * @author Audrey Loeffel
 */
public class SparseCudaBlas extends SparseNd4jBlas {
    @Override
    public void setMaxThreads(int num){

    }

    @Override
    public int getMaxThreads() {
        return 0;
    }

    @Override
    public int getBlasVendorId() {
        return 0;
    }
}
