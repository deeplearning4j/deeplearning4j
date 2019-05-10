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

import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.agrona.concurrent.SigIntBarrier;

import static java.lang.System.setProperty;
import static org.agrona.concurrent.UnsafeBuffer.DISABLE_BOUNDS_CHECKS_PROP_NAME;

/**
 * Created by agibsonccc on 9/22/16.
 */
public class LowLatencyMediaDriver {

    private LowLatencyMediaDriver() {}

    @SuppressWarnings("checkstyle:UncommentedMain")
    public static void main(final String... args) {
        MediaDriver.loadPropertiesFiles(args);

        setProperty(DISABLE_BOUNDS_CHECKS_PROP_NAME, "true");
        setProperty("aeron.mtu.length", "16384");
        setProperty("aeron.socket.so_sndbuf", "2097152");
        setProperty("aeron.socket.so_rcvbuf", "2097152");
        setProperty("aeron.rcv.initial.window.length", "2097152");

        final MediaDriver.Context ctx =
                        new MediaDriver.Context().threadingMode(ThreadingMode.DEDICATED).dirsDeleteOnStart(true)
                                        .termBufferSparseFile(false).conductorIdleStrategy(new BusySpinIdleStrategy())
                                        .receiverIdleStrategy(new BusySpinIdleStrategy())
                                        .senderIdleStrategy(new BusySpinIdleStrategy());

        try (MediaDriver ignored = MediaDriver.launch(ctx)) {
            new SigIntBarrier().await();

        }
    }

}
