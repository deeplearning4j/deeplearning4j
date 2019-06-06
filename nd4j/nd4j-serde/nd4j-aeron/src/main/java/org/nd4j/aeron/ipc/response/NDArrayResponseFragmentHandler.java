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

package org.nd4j.aeron.ipc.response;

import io.aeron.Aeron;
import io.aeron.logbuffer.FragmentHandler;
import io.aeron.logbuffer.Header;
import lombok.AllArgsConstructor;
import lombok.Builder;
import org.agrona.DirectBuffer;
import org.nd4j.aeron.ipc.AeronNDArrayPublisher;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.aeron.ipc.NDArrayHolder;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * A subscriber that listens for host
 * port pairs in the form of host:port.
 * These are meant to be aeron channels.
 *
 * Given an @link{NDArrayHolder} it will send
 * the ndarray to the designated channel by the subscriber.
 *
 * @author Adam Gibson
 */
@AllArgsConstructor
@Builder
public class NDArrayResponseFragmentHandler implements FragmentHandler {
    private NDArrayHolder holder;
    private Aeron.Context context;
    private Aeron aeron;
    private int streamId;

    /**
     * Callback for handling fragments of data being read from a log.
     *
     * @param buffer containing the data.
     * @param offset at which the data begins.
     * @param length of the data in bytes.
     * @param header representing the meta data for the data.
     */
    @Override
    public void onFragment(DirectBuffer buffer, int offset, int length, Header header) {
        if (buffer != null && length > 0) {
            ByteBuffer byteBuffer = buffer.byteBuffer().order(ByteOrder.nativeOrder());
            byteBuffer.position(offset);
            byte[] b = new byte[length];
            byteBuffer.get(b);
            String hostPort = new String(b);
            System.out.println("Host port " + hostPort + " offset " + offset + " length " + length);
            String[] split = hostPort.split(":");
            if (split == null || split.length != 3) {
                System.err.println("no host port stream found");
                return;
            }

            int port = Integer.parseInt(split[1]);
            int streamToPublish = Integer.parseInt(split[2]);
            String channel = AeronUtil.aeronChannel(split[0], port);
            INDArray arrGet = holder.get();
            AeronNDArrayPublisher publisher = AeronNDArrayPublisher.builder().streamId(streamToPublish).aeron(aeron)
                            .channel(channel).build();
            try {
                publisher.publish(arrGet);
            } catch (Exception e) {
                e.printStackTrace();
            }

            try {
                publisher.close();
            } catch (Exception e) {

            }
        }
    }
}
