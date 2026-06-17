/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.transforms.custom;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

import java.util.Collections;
import java.util.List;

/**
 * Device-to-host memory offload op for gradient checkpointing.
 *
 * <p>On CUDA backends this triggers an async DMA transfer of the input tensor
 * to pinned (page-locked) host memory, allowing GPU compute to overlap with
 * data movement during the forward pass.  On CPU or when the native op is not
 * registered the call degrades to a plain {@code dup()}.
 *
 * <p>Integer arguments:
 * <ol>
 *   <li>usePinned – 1 to allocate the host buffer as pinned memory, 0 otherwise</li>
 *   <li>streamIndex – CUDA stream index to use for the async transfer (0 = default)</li>
 * </ol>
 *
 * @author Adam Gibson
 * @see CheckpointPrefetchH2D
 */
@NoArgsConstructor
public class CheckpointOffloadD2H extends DynamicCustomOp {

    /**
     * Construct a D2H offload op for the given input tensor.
     * Uses default settings (no pinned memory, default stream).
     *
     * @param input the device tensor to offload to host
     */
    public CheckpointOffloadD2H(INDArray input) {
        addInputArgument(input);
        addIArgument(0, 0);
    }

    /**
     * Construct a D2H offload op with an explicit output buffer.
     *
     * @param input  the device tensor to offload
     * @param output pre-allocated host buffer to receive the data
     */
    public CheckpointOffloadD2H(INDArray input, INDArray output) {
        super(new INDArray[]{input}, wrapOrNull(output));
        addIArgument(0, 0);
    }

    /**
     * Construct a D2H offload op with pinned-memory control and an output buffer.
     *
     * @param input      the device tensor to offload
     * @param output     pre-allocated host buffer to receive the data (may be null)
     * @param usePinned  true to allocate the host buffer as pinned (page-locked) memory
     * @param streamIndex CUDA stream index for the async DMA transfer
     */
    public CheckpointOffloadD2H(INDArray input, INDArray output, boolean usePinned, int streamIndex) {
        super(new INDArray[]{input}, wrapOrNull(output));
        addIArgument(usePinned ? 1 : 0, streamIndex);
    }

    /**
     * Construct a D2H offload op for use inside a {@link SameDiff} graph.
     *
     * @param sameDiff the SameDiff instance
     * @param input    the variable whose value should be offloaded
     */
    public CheckpointOffloadD2H(SameDiff sameDiff, SDVariable input) {
        super(null, sameDiff, new SDVariable[]{input});
        addIArgument(0, 0);
    }

    @Override
    public String opName() {
        return "checkpoint_offload_d2h";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1,
                "Expected exactly 1 input datatype for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }
}
