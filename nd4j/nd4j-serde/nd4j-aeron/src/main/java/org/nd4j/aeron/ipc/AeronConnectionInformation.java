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

package org.nd4j.aeron.ipc;

import lombok.Builder;
import lombok.Data;

/**
 * Aeron connection information
 * pojo.
 * connectionHost represents the host for the media driver
 * connection host represents the port
 * stream represents the stream id to connect to
 * @author Adam Gibson
 */
@Data
@Builder
public class AeronConnectionInformation {
    private String connectionHost;
    private int connectionPort;
    private int streamId;

    /**
     * Traditional static generator method
     * @param connectionHost
     * @param connectionPort
     * @param streamId
     * @return
     */
    public static AeronConnectionInformation of(String connectionHost, int connectionPort, int streamId) {
        return AeronConnectionInformation.builder().connectionHost(connectionHost).connectionPort(connectionPort)
                        .streamId(streamId).build();
    }

    @Override
    public String toString() {
        return String.format("%s:%d:%d", connectionHost, connectionPort, streamId);
    }
}
