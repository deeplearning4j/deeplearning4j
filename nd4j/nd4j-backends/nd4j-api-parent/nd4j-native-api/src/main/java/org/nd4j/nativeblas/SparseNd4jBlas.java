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

package org.nd4j.nativeblas;

import org.nd4j.linalg.api.blas.Blas;

public abstract class SparseNd4jBlas implements Blas {

    public SparseNd4jBlas() {

    }

    /**
     * Returns the BLAS library vendor
     *
     * @return the BLAS library vendor
     */
    @Override
    public Vendor getBlasVendor() {
        int vendor = getBlasVendorId();
        boolean isUnknowVendor = ((vendor > Vendor.values().length - 1) || (vendor <= 0));
        if (isUnknowVendor) {
            return Vendor.UNKNOWN;
        }
        return Vendor.values()[vendor];
    }
}
