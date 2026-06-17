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

/**
 * One-time initialization report for DSP diagnostics.
 * Contains model metadata and DSP configuration at attach time.
 */
@Data
@NoArgsConstructor
public class DspDiagnosticsInitReport implements Persistable {

    public static final String TYPE_ID = DspDiagnosticsReport.TYPE_ID;
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private String sessionID;
    private String workerID;
    private long timeStamp;

    private int variableCount;
    private int opCount;
    private long paramCount;
    private List<String> paramNames;
    private String configuredExecutionMode;
    private List<String> requestedOutputs;
    private String backend;
    private String dataType;
    private String hostname;

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
            throw new RuntimeException("Failed to encode DspDiagnosticsInitReport", e);
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
            DspDiagnosticsInitReport r = MAPPER.readValue(decode, DspDiagnosticsInitReport.class);
            this.sessionID = r.sessionID;
            this.workerID = r.workerID;
            this.timeStamp = r.timeStamp;
            this.variableCount = r.variableCount;
            this.opCount = r.opCount;
            this.paramCount = r.paramCount;
            this.paramNames = r.paramNames;
            this.configuredExecutionMode = r.configuredExecutionMode;
            this.requestedOutputs = r.requestedOutputs;
            this.backend = r.backend;
            this.dataType = r.dataType;
            this.hostname = r.hostname;
        } catch (Exception e) {
            throw new RuntimeException("Failed to decode DspDiagnosticsInitReport", e);
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
