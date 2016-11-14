package org.nd4j.aeron.ipc;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import org.agrona.CloseHelper;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
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
public class NdArrayIpcTest {
    private MediaDriver mediaDriver;
    private static Logger log = LoggerFactory.getLogger(NdArrayIpcTest.class);
    private Aeron.Context ctx;

    @Before
    public void before() {
        final MediaDriver.Context ctx = new MediaDriver.Context()
                .threadingMode(ThreadingMode.DEDICATED)
                .dirsDeleteOnStart(true)
                .termBufferSparseFile(false)
                .conductorIdleStrategy(new BusySpinIdleStrategy())
                .receiverIdleStrategy(new BusySpinIdleStrategy())
                .senderIdleStrategy(new BusySpinIdleStrategy());
        mediaDriver = MediaDriver.launchEmbedded(ctx);
        System.out.println("Using media driver directory " + mediaDriver.aeronDirectoryName());
        System.out.println("Launched media driver");
    }

    @After
    public void after() {
        CloseHelper.quietClose(mediaDriver);
    }


    @Test
    public void testMultiThreadedIpc() throws Exception {
        //LowLatencyMediaDriver.main(new String[]{});

        ExecutorService executorService = Executors.newFixedThreadPool(4);
        INDArray arr = Nd4j.scalar(1.0);


        final AtomicBoolean running = new AtomicBoolean(true);


        AeronNDArraySubscriber subscriber = AeronNDArraySubscriber.builder().streamId(10)
                .ctx(getContext()).channel("aeron:udp?endpoint=localhost:40123")
                .running(running)
                .ndArrayCallback(new NDArrayCallback() {
                    @Override
                    public void onNDArrayPartial(INDArray arr, long idx, int... dimensions) {

                    }

                    @Override
                    public void onNDArray(INDArray arr) {
                        System.out.println(arr);
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


        Thread.sleep(10000);

        for(int i = 0; i< 10 && running.get(); i++) {
           executorService.execute(() -> {
               try {
                   log.info("About to send array.");
                   try(AeronNDArrayPublisher publisher =   AeronNDArrayPublisher.builder().streamId(10)
                           .ctx(getContext()).channel("aeron:udp?endpoint=localhost:40123")
                           .build()) {
                       publisher.publish(arr);
                       log.info("Sent array in pool");
                       Thread.sleep(10);
                   }

               } catch (Exception e) {
                   e.printStackTrace();
               }
           });

        }

        Thread.sleep(100000);



        assertFalse(running.get());
    }

    @Test
    public void testIpc() throws Exception {
        //LowLatencyMediaDriver.main(new String[]{});
        INDArray arr = Nd4j.scalar(1.0);


        final AtomicBoolean running = new AtomicBoolean(true);


        AeronNDArraySubscriber subscriber = AeronNDArraySubscriber.builder().streamId(10)
                .ctx(getContext()).channel("aeron:udp?endpoint=localhost:40123")
                .running(running)
                .ndArrayCallback(new NDArrayCallback() {
                    @Override
                    public void onNDArrayPartial(INDArray arr, long idx, int... dimensions) {

                    }

                    @Override
                    public void onNDArray(INDArray arr) {
                        System.out.println(arr);
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

        AeronNDArrayPublisher publisher =   AeronNDArrayPublisher.builder().streamId(10)
                .ctx(getContext()).channel("aeron:udp?endpoint=localhost:40123")
                .build();
        for(int i = 0; i< 10 && running.get(); i++) {
            publisher.publish(arr);
            Thread.sleep(10);
        }
        publisher.close();
        Thread.sleep(10000);
        assertFalse(running.get());

    }


    private Aeron.Context getContext() {
       if(ctx == null) ctx = new Aeron.Context().publicationConnectionTimeout(-1).availableImageHandler(AeronUtil::printAvailableImage)
                .unavailableImageHandler(AeronUtil::printUnavailableImage)
                .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                .errorHandler(e -> log.error(e.toString(), e));
        return ctx;
    }
}
