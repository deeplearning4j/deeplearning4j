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

package org.nd4j.linalg.framework.device;

import lombok.Builder;
import lombok.Data;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Full diagnostic report snapshot.
 *
 * @author ND4J Team
 */
@Data
@Builder
public class TransferReport {
    /**
     * Timestamp in milliseconds
     */
    private long timestampMs;

    /**
     * Per-variable statistics
     */
    @Builder.Default
    private Map<String, TransferStats> perVariableStats = new HashMap<>();

    /**
     * Total transfer count
     */
    private long totalTransferCount;

    /**
     * Total bytes transferred
     */
    private long totalBytes;

    /**
     * Window duration in milliseconds
     */
    private long windowDurationMs;

    /**
     * Recent events (last N events from ring buffer)
     */
    @Builder.Default
    private List<TransferEvent> recentEvents = new ArrayList<>();

    /**
     * Generate a human-readable summary.
     */
    public String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Transfer Report [");
        sb.append(timestampMs).append("ms] ");
        sb.append("Total: ").append(totalTransferCount).append(" transfers, ");
        sb.append(totalBytes).append(" bytes");
        if (windowDurationMs > 0) {
            double bandwidth = (totalBytes * 1000.0) / windowDurationMs;
            sb.append(String.format(", Bandwidth: %.2f MB/s", bandwidth / (1024.0 * 1024.0)));
        }
        if (!perVariableStats.isEmpty()) {
            sb.append("\nPer-variable:");
            for (Map.Entry<String, TransferStats> entry : perVariableStats.entrySet()) {
                TransferStats stats = entry.getValue();
                sb.append("\n  ").append(entry.getKey())
                    .append(": ").append(stats.getTotalTransfers()).append(" transfers, ")
                    .append(stats.getTotalBytes()).append(" bytes");
            }
        }
        return sb.toString();
    }
}
