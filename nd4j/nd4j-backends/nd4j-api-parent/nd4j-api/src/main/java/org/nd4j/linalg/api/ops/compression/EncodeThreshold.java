/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.compression;

import lombok.NonNull;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Arrays;
import java.util.List;

/**
 * Sparse threshold encoding op wrapper. Used in gradients sharing.
 * @author raver119@gmail.com
 */
public class EncodeThreshold extends DynamicCustomOp {
    protected float threshold = 1e-3f;
    protected int boundary = Integer.MAX_VALUE;

    public EncodeThreshold() {
        //
    }

    public EncodeThreshold(@NonNull INDArray updates, float threshold) {
        this(updates, threshold, Integer.MAX_VALUE);
    }

    public EncodeThreshold(@NonNull INDArray updates, @NonNull INDArray encoded, float threshold, @NonNull Integer boundary) {
        this(updates, threshold, boundary);

        addOutputArgument(updates, encoded);
    }

    public EncodeThreshold(@NonNull INDArray updates, float threshold, @NonNull Integer boundary) {
        addInputArgument(updates);

        addTArgument(threshold);
        addIArgument(boundary.intValue());

        this.threshold = threshold;
        this.boundary = boundary;
    }

    @Override
    public String opName() {
        return "encode_threshold";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        return Arrays.asList(inputArguments.get(0).dataType(), DataType.INT32);
    }
}
