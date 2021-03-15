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

package org.nd4j.aeron.ipc;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import lombok.extern.slf4j.Slf4j;
import org.agrona.CloseHelper;
import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.concurrent.NotThreadSafe;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.Assert.assertFalse;

@Slf4j
@NotThreadSafe
@Ignore("Tests are too flaky")
public class LargeNdArrayIpcTest extends BaseND4JTest {
    private MediaDriver mediaDriver;
    private Aeron.Context ctx;
    private String channel = "aeron:udp?endpoint=localhost:" + (40123 + new java.util.Random().nextInt(130));
    private int streamId = 10;
    private int length = (int) 1e7;

    @Override
    public long getTimeoutMilliseconds() {
        return 180000L;
    }

    @Before
    public void before() {
        if(isIntegrationTests()) {
            //MediaDriver.loadPropertiesFile("aeron.properties");
            MediaDriver.Context ctx = AeronUtil.getMediaDriverContext(length);
            mediaDriver = MediaDriver.launchEmbedded(ctx);
            System.out.println("Using media driver directory " + mediaDriver.aeronDirectoryName());
            System.out.println("Launched media driver");
        }
    }

    @After
    public void after() {
        if(isIntegrationTests()) {
            CloseHelper.quietClose(mediaDriver);
        }
    }

    @Test
    @Ignore
    public void testMultiThreadedIpcBig() throws Exception {
        skipUnlessIntegrationTests();   //Long-running test - don't run as part of unit tests by default

        int length = (int) 1e7;
        INDArray arr = Nd4j.ones(length);
        AeronNDArrayPublisher publisher;
        ctx = new Aeron.Context()
                .driverTimeoutMs(1000000).availableImageHandler(AeronUtil::printAvailableImage)
                        .unavailableImageHandler(AeronUtil::printUnavailableImage)
                        .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveIntervalNs(1000000)
                        .errorHandler(err -> err.printStackTrace());

        final AtomicBoolean running = new AtomicBoolean(true);
        Aeron aeron = Aeron.connect(ctx);
        int numSubscribers = 1;
        AeronNDArraySubscriber[] subscribers = new AeronNDArraySubscriber[numSubscribers];
        for (int i = 0; i < numSubscribers; i++) {
            AeronNDArraySubscriber subscriber = AeronNDArraySubscriber.builder().streamId(streamId).ctx(getContext())
                            .channel(channel).aeron(aeron).running(running).ndArrayCallback(new NDArrayCallback() {
                                /**
                                 * A listener for ndarray message
                                 *
                                 * @param message the message for the callback
                                 */
                                @Override
                                public void onNDArrayMessage(NDArrayMessage message) {
                                    running.set(false);
                                }

                                @Override
                                public void onNDArrayPartial(INDArray arr, long idx, int... dimensions) {

                            }

                                @Override
                                public void onNDArray(INDArray arr) {
                                    running.set(false);
                                }
                            }).build();


            Thread t = new Thread(() -> {
                try {
                    subscriber.launch();
                } catch (Exception e) {
                    log.error("",e);
                }

            });

            t.start();

            subscribers[i] = subscriber;
        }

        Thread.sleep(10000);

        publisher = AeronNDArrayPublisher.builder().publishRetryTimeOut(300000).streamId(streamId).channel(channel)
                        .aeron(aeron).build();


        for (int i = 0; i < 1 && running.get(); i++) {
            log.info("About to send array.");
            publisher.publish(arr);
            log.info("Sent array");

        }

        Thread.sleep(30000);



        for (int i = 0; i < numSubscribers; i++)
            CloseHelper.close(subscribers[i]);
        CloseHelper.close(aeron);
        CloseHelper.close(publisher);
        assertFalse(running.get());
    }



    private Aeron.Context getContext() {
        if (ctx == null)
            ctx = new Aeron.Context().driverTimeoutMs(1000000)
                            .availableImageHandler(AeronUtil::printAvailableImage)
                            .unavailableImageHandler(AeronUtil::printUnavailableImage)
                            .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveIntervalNs(100000)
                            .errorHandler(err -> err.printStackTrace());
        return ctx;
    }
}
