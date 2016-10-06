package org.nd4j.parameterserver.background;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import io.aeron.driver.ThreadingMode;
import lombok.extern.slf4j.Slf4j;
import org.agrona.concurrent.BusySpinIdleStrategy;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.aeron.ipc.AeronUtil;

import java.io.IOException;

/**
 * Created by agibsonccc on 10/5/16.
 */
@Slf4j
public class RemoteParameterServerClientTests {
    private int parameterLength = 1000;
    private Aeron.Context ctx;
    private MediaDriver mediaDriver;

    @Before
    public void before() throws Exception {
        final MediaDriver.Context ctx = new MediaDriver.Context()
                .threadingMode(ThreadingMode.DEDICATED)
                .dirsDeleteOnStart(true)
                .termBufferSparseFile(false)
                .conductorIdleStrategy(new BusySpinIdleStrategy())
                .receiverIdleStrategy(new BusySpinIdleStrategy())
                .senderIdleStrategy(new BusySpinIdleStrategy());

        mediaDriver = MediaDriver.launchEmbedded(ctx);


        Thread t = new Thread(() -> {
            try {
                BackgroundDaemonStarter.startMaster(parameterLength,mediaDriver.aeronDirectoryName());
            } catch (IOException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });

        t.start();
        log.info("Started master");
        Thread t2 = new Thread(() -> {
            try {
                BackgroundDaemonStarter.startSlave(parameterLength,mediaDriver.aeronDirectoryName());
            } catch (IOException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
        t2.start();
        log.info("Started slave");

        Thread.sleep(10000);
    }


    @After
    public void after() throws Exception {
        mediaDriver.close();
    }

    @Test
    public void remoteTests() {

    }


    private Aeron.Context getContext() {
        if(ctx == null)
            ctx = new Aeron.Context().publicationConnectionTimeout(-1)
                    .availableImageHandler(AeronUtil::printAvailableImage)
                    .unavailableImageHandler(AeronUtil::printUnavailableImage)
                    .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                    .errorHandler(e -> log.error(e.toString(), e));
        return ctx;
    }

}
