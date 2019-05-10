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
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import lombok.extern.slf4j.Slf4j;
import org.agrona.CloseHelper;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.aeron.ipc.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 10/3/16.
 */
@Slf4j
public class AeronNDArrayResponseTest {
    private MediaDriver mediaDriver;

    @Before
    public void before() {
        final MediaDriver.Context ctx =
                        new MediaDriver.Context().threadingMode(ThreadingMode.SHARED).dirsDeleteOnStart(true)
                                        .termBufferSparseFile(false).conductorIdleStrategy(new BusySpinIdleStrategy())
                                        .receiverIdleStrategy(new BusySpinIdleStrategy())
                                        .senderIdleStrategy(new BusySpinIdleStrategy());
        mediaDriver = MediaDriver.launchEmbedded(ctx);
        System.out.println("Using media driver directory " + mediaDriver.aeronDirectoryName());
        System.out.println("Launched media driver");
    }


    @Test
    public void testResponse() throws Exception {
        int streamId = 10;
        int responderStreamId = 11;
        String host = "127.0.0.1";
        Aeron.Context ctx = new Aeron.Context().publicationConnectionTimeout(-1)
                        .availableImageHandler(AeronUtil::printAvailableImage)
                        .unavailableImageHandler(AeronUtil::printUnavailableImage)
                        .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                        .errorHandler(e -> log.error(e.toString(), e));

        int baseSubscriberPort = 40123 + new java.util.Random().nextInt(1000);

        Aeron aeron = Aeron.connect(ctx);
        AeronNDArrayResponder responder =
                        AeronNDArrayResponder.startSubscriber(aeron, host, baseSubscriberPort + 1, new NDArrayHolder() {
                            /**
                             * Set the ndarray
                             *
                             * @param arr the ndarray for this holder
                             *            to use
                             */
                            @Override
                            public void setArray(INDArray arr) {

                        }

                            /**
                            * The number of updates
                            * that have been sent to this older.
                            *
                            * @return
                            */
                            @Override
                            public int totalUpdates() {
                                return 1;
                            }

                            /**
                             * Retrieve an ndarray
                             *
                             * @return
                             */
                            @Override
                            public INDArray get() {
                                return Nd4j.scalar(1.0);
                            }

                            /**
                             * Retrieve a partial view of the ndarray.
                             * This method uses tensor along dimension internally
                             * Note this will call dup()
                             *
                             * @param idx        the index of the tad to get
                             * @param dimensions the dimensions to use
                             * @return the tensor along dimension based on the index and dimensions
                             * from the master array.
                             */
                            @Override
                            public INDArray getTad(int idx, int... dimensions) {
                                return Nd4j.scalar(1.0);
                            }
                        }

                                        , responderStreamId);

        AtomicInteger count = new AtomicInteger(0);
        AtomicBoolean running = new AtomicBoolean(true);
        AeronNDArraySubscriber subscriber =
                        AeronNDArraySubscriber.startSubscriber(aeron, host, baseSubscriberPort, new NDArrayCallback() {
                            /**
                             * A listener for ndarray message
                             *
                             * @param message the message for the callback
                             */
                            @Override
                            public void onNDArrayMessage(NDArrayMessage message) {
                                count.incrementAndGet();

                            }

                            @Override
                            public void onNDArrayPartial(INDArray arr, long idx, int... dimensions) {
                                count.incrementAndGet();
                            }

                            @Override
                            public void onNDArray(INDArray arr) {
                                count.incrementAndGet();
                            }
                        }, streamId, running);

        int expectedResponses = 10;
        HostPortPublisher publisher = HostPortPublisher.builder().aeron(aeron)
                        .uriToSend(host + String.format(":%d:", baseSubscriberPort) + streamId)
                        .channel(AeronUtil.aeronChannel(host, baseSubscriberPort + 1)).streamId(responderStreamId)
                        .build();



        for (int i = 0; i < expectedResponses; i++) {
            publisher.send();
        }


        Thread.sleep(60000);



        assertEquals(expectedResponses, count.get());

        System.out.println("After");

        CloseHelper.close(responder);
        CloseHelper.close(subscriber);
        CloseHelper.close(publisher);
        CloseHelper.close(aeron);

    }



}
