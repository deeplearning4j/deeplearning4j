/*
 *  ******************************************************************************
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

package org.deeplearning4j.ui.model.stats.dsp;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.core.storage.Persistable;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;

/**
 * Per-iteration DSP (Dynamic Shape Plan) diagnostics report.
 * JSON-encoded Persistable carrying plan state, segment info, replay state,
 * op histogram, memory timeline, and DOT graph.
 */
@Data
@NoArgsConstructor
public class DspDiagnosticsReport implements Persistable {

    public static final String TYPE_ID = "DspDiagnostics";
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private String sessionID;
    private String workerID;
    private long timeStamp;
    private int iteration;

    // Plan-level state
    private int numSlots;
    private int numSegments;
    private String planPhase;
    private String graphNodePhase;
    private String executionMode;

    // Segments with replay state
    private List<SegmentSnapshot> segments;

    // Op histogram
    private Map<String, Integer> opHistogram;

    // Risky ops
    private List<SlotSnapshot> riskyOps;
    private int unfrozenOpCount;

    // Memory timeline
    private List<MemoryEventSnapshot> memoryTimeline;

    // DOT graph
    private String dotGraph;

    // Timing
    private long iterationTimeMs;

    // Device placement
    private Map<Integer, Integer> deviceSlotCounts;

    @Override
    public String getTypeID() {
        return TYPE_ID;
    }

    @Override
    public int encodingLengthBytes() {
        return encode().length;
    }

    @Override
    public byte[] encode() {
        try {
            return MAPPER.writeValueAsBytes(this);
        } catch (Exception e) {
            throw new RuntimeException("Failed to encode DspDiagnosticsReport", e);
        }
    }

    @Override
    public void encode(ByteBuffer buffer) {
        buffer.put(encode());
    }

    @Override
    public void encode(OutputStream outputStream) throws IOException {
        outputStream.write(encode());
    }

    @Override
    public void decode(byte[] decode) {
        try {
            DspDiagnosticsReport r = MAPPER.readValue(decode, DspDiagnosticsReport.class);
            this.sessionID = r.sessionID;
            this.workerID = r.workerID;
            this.timeStamp = r.timeStamp;
            this.iteration = r.iteration;
            this.numSlots = r.numSlots;
            this.numSegments = r.numSegments;
            this.planPhase = r.planPhase;
            this.graphNodePhase = r.graphNodePhase;
            this.executionMode = r.executionMode;
            this.segments = r.segments;
            this.opHistogram = r.opHistogram;
            this.riskyOps = r.riskyOps;
            this.unfrozenOpCount = r.unfrozenOpCount;
            this.memoryTimeline = r.memoryTimeline;
            this.dotGraph = r.dotGraph;
            this.iterationTimeMs = r.iterationTimeMs;
            this.deviceSlotCounts = r.deviceSlotCounts;
        } catch (Exception e) {
            throw new RuntimeException("Failed to decode DspDiagnosticsReport", e);
        }
    }

    @Override
    public void decode(ByteBuffer buffer) {
        byte[] arr = new byte[buffer.remaining()];
        buffer.get(arr);
        decode(arr);
    }

    @Override
    public void decode(InputStream inputStream) throws IOException {
        decode(IOUtils.toByteArray(inputStream));
    }

}
