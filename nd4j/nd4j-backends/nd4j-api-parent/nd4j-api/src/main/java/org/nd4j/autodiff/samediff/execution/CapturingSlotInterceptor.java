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

package org.nd4j.autodiff.samediff.execution;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * A {@link SlotOutputInterceptor} that captures {@code dup()} snapshots of slot outputs.
 * Supports optional filtering by variable name and capture interval.
 *
 * <p>Captured arrays are stored in two views:
 * <ul>
 *   <li>{@code captured}: stepIdx → {varName → snapshot} — full history by step</li>
 *   <li>{@code byName}: varName → latest snapshot — most recent value per variable</li>
 * </ul>
 */
public class CapturingSlotInterceptor implements SlotOutputInterceptor {

    private final Map<Integer, Map<String, INDArray>> captured = new LinkedHashMap<>();
    private final Map<String, INDArray> byName = new LinkedHashMap<>();

    private Set<String> filterVarNames;
    private int captureInterval = 1;

    /**
     * Only capture outputs for variables in this set. Null means capture all.
     */
    public CapturingSlotInterceptor filterVarNames(Set<String> varNames) {
        this.filterVarNames = varNames;
        return this;
    }

    /**
     * Capture every Nth step only. Default is 1 (every step).
     */
    public CapturingSlotInterceptor captureInterval(int interval) {
        this.captureInterval = Math.max(1, interval);
        return this;
    }

    @Override
    public void onSlotOutput(int stepIdx, String opName, String outputVarName, INDArray output, int outputSlotIdx) {
        if (output == null) return;
        if (captureInterval > 1 && stepIdx % captureInterval != 0) return;
        if (filterVarNames != null && !filterVarNames.contains(outputVarName)) return;

        INDArray snapshot = output.dup();
        captured.computeIfAbsent(stepIdx, k -> new LinkedHashMap<>()).put(outputVarName, snapshot);
        byName.put(outputVarName, snapshot);
    }

    /**
     * Returns all captured snapshots indexed by step and variable name.
     */
    public Map<Integer, Map<String, INDArray>> getCaptured() {
        return captured;
    }

    /**
     * Returns the latest snapshot for each captured variable name.
     */
    public Map<String, INDArray> getByName() {
        return byName;
    }

    /**
     * Closes all captured arrays and clears internal state.
     */
    public void clear() {
        for (Map<String, INDArray> stepMap : captured.values()) {
            for (INDArray arr : stepMap.values()) {
                if (!arr.wasClosed()) arr.close();
            }
        }
        captured.clear();
        byName.clear();
    }
}
