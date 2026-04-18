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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Memory management utilities for SameDiff models.
 *
 * <p>Addresses the "closeable poisoning" problem: {@code SameDiff.directExecHelper()}
 * calls {@code setCloseable(false)} on all placeholders, which sets
 * {@code dataBuffer.setConstant(true)} — marking the buffer as constant and
 * preventing {@code close()} from freeing memory. {@link #safeClose(INDArray)}
 * undoes this poisoning before closing.</p>
 */
@Slf4j
public final class SameDiffMemoryUtils {

    private SameDiffMemoryUtils() {}

    /**
     * Safely close an INDArray, undoing any "closeable poisoning" from SameDiff execution.
     *
     * <p>{@code SameDiff.directExecHelper()} calls {@code setCloseable(false)} on all
     * placeholder arrays, which internally sets {@code dataBuffer.setConstant(true)}.
     * This prevents {@code close()} from actually freeing the GPU memory. This method
     * calls {@code setCloseable(true)} first to undo the poisoning, then closes.</p>
     *
     * @param arr the array to close, may be null
     */
    public static void safeClose(INDArray arr) {
        if (arr == null) {
            return;
        }
        try {
            if (arr.wasClosed()) {
                return;
            }
            arr.setCloseable(true);
            arr.close();
        } catch (Exception e) {
            log.debug("Exception during safeClose: {}", e.getMessage());
        }
    }

    /**
     * Free all constant and variable arrays from a SameDiff model.
     *
     * <p>Iterates through the model's constant and variable array holders,
     * removes each array, undoes the constant flag, and closes it.
     * This is useful for freeing model weights after they are no longer needed
     * (e.g., freeing a vision encoder after encoding is complete).</p>
     *
     * @param model the SameDiff model whose arrays should be freed
     * @return the number of arrays freed
     */
    public static int freeModelArrays(SameDiff model) {
        if (model == null) {
            return 0;
        }

        int freedCount = 0;

        // Free constant arrays
        freedCount += freeArrayHolder(model.getConstantArrays(), "constant");

        // Free variable arrays
        freedCount += freeArrayHolder(model.getVariablesArrays(), "variable");

        log.info("Freed {} model arrays", freedCount);
        return freedCount;
    }

    private static int freeArrayHolder(ArrayHolder holder, String holderType) {
        if (holder == null) {
            return 0;
        }

        Collection<String> names = holder.arrayNames();
        if (names == null || names.isEmpty()) {
            return 0;
        }

        // Copy names to avoid ConcurrentModificationException
        List<String> nameList = new ArrayList<>(names);
        int freedCount = 0;

        for (String name : nameList) {
            try {
                INDArray arr = holder.removeArray(name);
                if (arr != null && !arr.wasClosed()) {
                    if (arr.data() != null) {
                        arr.data().setConstant(false);
                    }
                    arr.setCloseable(true);
                    arr.close();
                    freedCount++;
                }
            } catch (Exception e) {
                log.debug("Exception freeing {} array '{}': {}", holderType, name, e.getMessage());
            }
        }

        return freedCount;
    }

    /**
     * Trim CUDA memory pools on all devices.
     * Syncs pending cudaFreeAsync calls and releases physical memory back to the OS.
     */
    public static void trimAllDevicePools() {
        try {
            var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int d = 0; d < numDevices; d++) {
                nativeOps.trimMemoryPoolOnStream(d, null);
            }
        } catch (Exception e) {
            log.debug("Failed to trim memory pools: {}", e.getMessage());
        }
    }
}
