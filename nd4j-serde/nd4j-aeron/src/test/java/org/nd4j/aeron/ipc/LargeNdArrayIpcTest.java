package org.nd4j.aeron.ipc;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.status.SystemCounterDescriptor;
import lombok.extern.slf4j.Slf4j;
import org.agrona.CloseHelper;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.aeron.util.AeronStat;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.Assert.assertFalse;

/**
 * Created by agibsonccc on 9/22/16.
 */
@Slf4j
public class LargeNdArrayIpcTest {
    private MediaDriver mediaDriver;
    private Aeron.Context ctx;
    private String channel = "aeron:udp?endpoint=localhost:40123";
    private int streamId = 10;
    private  int length = (int) 1e7;

    @Before
    public void before() {
        //MediaDriver.loadPropertiesFile("aeron.properties");
        MediaDriver.Context ctx = AeronUtil.getMediaDriverContext(length);
        mediaDriver = MediaDriver.launchEmbedded(ctx);
        System.out.println("Using media driver directory " + mediaDriver.aeronDirectoryName());
        System.out.println("Launched media driver");
    }

    @After
    public void after() {
        CloseHelper.quietClose(mediaDriver);
    }

    @Test
    public void testMultiThreadedIpcBig() throws Exception {
        int length = (int) 1e7;
        INDArray arr = Nd4j.ones(length);


        final AtomicBoolean running = new AtomicBoolean(true);
        Aeron aeron = Aeron.connect(getContext());
        int numSubscribers = 1;
        AeronNDArraySubscriber[] subscribers = new AeronNDArraySubscriber[numSubscribers];
        for(int i = 0; i < numSubscribers; i++) {
            AeronNDArraySubscriber subscriber = AeronNDArraySubscriber.builder()
                    .streamId(streamId)
                    .ctx(getContext()).channel(channel).aeron(aeron)
                    .running(running)
                    .ndArrayCallback(new NDArrayCallback() {
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
                    e.printStackTrace();
                }

            });

            t.start();

            subscribers[i] = subscriber;
        }

        Thread.sleep(10000);

        AeronNDArrayPublisher publisher =   AeronNDArrayPublisher
                .builder().publishRetryTimeOut(100)
                .streamId(streamId)
                .channel(channel).aeron(aeron)
                .build();


        for(int i = 0; i< 10 && running.get(); i++) {
            log.info("About to send array.");
            publisher.publish(arr);
            log.info("Sent array");

        }

        Thread t = new Thread(() -> {
            System.setProperty("aeron.dir",mediaDriver.aeronDirectoryName());
            try {
                AeronStat.main(new String[]{});
            } catch (Exception e) {
                e.printStackTrace();
            }

        });

        t.start();

        Thread.sleep(10000);



        for(int i = 0; i < numSubscribers; i++)
            CloseHelper.close(subscribers[i]);
        CloseHelper.close(aeron);
        CloseHelper.close(publisher);
        assertFalse(running.get());
    }





    private Aeron.Context getContext() {
        if(ctx == null) ctx = new Aeron.Context().publicationConnectionTimeout(-1)
                .availableImageHandler(AeronUtil::printAvailableImage)
                .unavailableImageHandler(AeronUtil::printUnavailableImage)
                .aeronDirectoryName(mediaDriver.aeronDirectoryName())
                .keepAliveInterval(10000)
                .errorHandler(err -> err.printStackTrace());
        return ctx;
    }
}
