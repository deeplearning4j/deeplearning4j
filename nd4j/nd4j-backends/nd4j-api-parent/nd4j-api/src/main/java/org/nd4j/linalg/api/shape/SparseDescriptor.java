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

package org.nd4j.linalg.api.shape;

import java.util.Arrays;

/**
 * @author Audrey Loeffel
 */
public class SparseDescriptor {

    int[] flags;
    long[] sparseOffsets;
    int[] hiddenDimension;
    int underlyingRank;

    public SparseDescriptor(int[] flags, long[] sparseOffsets, int[] hiddenDimension, int underlyingRank) {
        this.flags = Arrays.copyOf(flags, flags.length);
        this.sparseOffsets = Arrays.copyOf(sparseOffsets, sparseOffsets.length);
        this.hiddenDimension = Arrays.copyOf(hiddenDimension, hiddenDimension.length);
        this.underlyingRank = underlyingRank;
    }

    @Override
    public int hashCode() {
        int result = underlyingRank;
        result = 31 * result + Arrays.hashCode(flags);
        result = 31 * result + Arrays.hashCode(sparseOffsets);
        result = 31 * result + Arrays.hashCode(hiddenDimension);
        return result;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        SparseDescriptor that = (SparseDescriptor) o;

        if (!Arrays.equals(flags, that.flags))
            return false;
        if (!Arrays.equals(sparseOffsets, that.sparseOffsets))
            return false;
        if (!Arrays.equals(hiddenDimension, that.hiddenDimension))
            return false;
        return underlyingRank == that.underlyingRank;
    }

    @Override
    public String toString() {

        StringBuilder builder = new StringBuilder();

        builder.append(flags.length).append(",").append(Arrays.toString(flags)).append(",").append(sparseOffsets.length)
                        .append(",").append(Arrays.toString(sparseOffsets)).append(",").append(hiddenDimension.length)
                        .append(",").append(Arrays.toString(hiddenDimension)).append(",").append(underlyingRank);

        String result = builder.toString().replaceAll("\\]", "").replaceAll("\\[", "");
        result = "[" + result + "]";

        return result;
    }
}
