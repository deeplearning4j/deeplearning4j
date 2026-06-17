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
 * Host-to-device prefetch op for gradient checkpointing.
 *
 * <p>On CUDA backends this triggers an async DMA transfer of a host-resident
 * (optionally pinned) tensor back to the device, allowing the backward pass to
 * overlap data movement with earlier gradient computations.  On CPU or when the
 * native op is not registered the call degrades to a plain {@code dup()}.
 *
 * <p>Integer arguments:
 * <ol>
 *   <li>streamIndex – CUDA stream index to use for the async transfer (0 = default)</li>
 * </ol>
 *
 * @author Adam Gibson
 * @see CheckpointOffloadD2H
 */
@NoArgsConstructor
public class CheckpointPrefetchH2D extends DynamicCustomOp {

    /**
     * Construct an H2D prefetch op for the given host tensor.
     * Uses the default CUDA stream (index 0).
     *
     * @param input the host tensor to prefetch to device
     */
    public CheckpointPrefetchH2D(INDArray input) {
        addInputArgument(input);
        addIArgument(0);
    }

    /**
     * Construct an H2D prefetch op with an explicit output buffer.
     *
     * @param input  the host tensor to prefetch
     * @param output pre-allocated device buffer to receive the data (may be null)
     */
    public CheckpointPrefetchH2D(INDArray input, INDArray output) {
        super(new INDArray[]{input}, wrapOrNull(output));
        addIArgument(0);
    }

    /**
     * Construct an H2D prefetch op with an explicit stream index and output buffer.
     *
     * @param input       the host tensor to prefetch
     * @param output      pre-allocated device buffer to receive the data (may be null)
     * @param streamIndex CUDA stream index for the async DMA transfer
     */
    public CheckpointPrefetchH2D(INDArray input, INDArray output, int streamIndex) {
        super(new INDArray[]{input}, wrapOrNull(output));
        addIArgument(streamIndex);
    }

    /**
     * Construct an H2D prefetch op for use inside a {@link SameDiff} graph.
     *
     * @param sameDiff the SameDiff instance
     * @param input    the variable whose value should be prefetched to device
     */
    public CheckpointPrefetchH2D(SameDiff sameDiff, SDVariable input) {
        super(null, sameDiff, new SDVariable[]{input});
        addIArgument(0);
    }

    @Override
    public String opName() {
        return "checkpoint_prefetch_h2d";
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        Preconditions.checkState(dataTypes != null && dataTypes.size() == 1,
                "Expected exactly 1 input datatype for %s, got %s", getClass(), dataTypes);
        return Collections.singletonList(dataTypes.get(0));
    }
}
