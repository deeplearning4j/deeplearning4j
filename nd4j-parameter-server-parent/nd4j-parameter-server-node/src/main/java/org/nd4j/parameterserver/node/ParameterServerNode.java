package org.nd4j.parameterserver.node;

import io.aeron.Aeron;
import io.aeron.driver.MediaDriver;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.agrona.CloseHelper;
import org.nd4j.aeron.ipc.AeronUtil;
import org.nd4j.parameterserver.ParameterServerSubscriber;
import org.nd4j.parameterserver.status.play.InMemoryStatusStorage;
import org.nd4j.parameterserver.status.play.StatusServer;
import play.server.Server;

/**
 * Integrated node for running
 * the parameter server.
 * This includes the status server,
 * media driver, and parameter server subscriber
 *
 * @author Adam Gibson
 */
@Slf4j
@NoArgsConstructor
@Data
public class ParameterServerNode {
    private Server server;
    private ParameterServerSubscriber subscriber;
    private MediaDriver mediaDriver;
    private Aeron aeron;
    private int statusPort;

    /**
     *
     * @param mediaDriver the media driver to sue for communication
     * @param statusPort the port for the server status
     */
    public ParameterServerNode(MediaDriver mediaDriver,int statusPort) {
        this.mediaDriver = mediaDriver;
        this.statusPort = statusPort;

    }


    /**
     * Pass in the media driver used for communication
     * and a defualt status port of 9000
     * @param mediaDriver
     */
    public ParameterServerNode(MediaDriver mediaDriver) {
        this(mediaDriver,9000);
    }

    /**
     * Run this node with the given args
     * These args are the same ones
     * that a {@link ParameterServerSubscriber} takes
     * @param args the arguments for the {@link ParameterServerSubscriber}
     */
    public void runMain(String[] args) {
        server = StatusServer.startServer(new InMemoryStatusStorage(),statusPort);
        if(mediaDriver == null)
            mediaDriver = MediaDriver.launchEmbedded();
        log.info("Started media driver with aeron directory " + mediaDriver.aeronDirectoryName());
        subscriber = new ParameterServerSubscriber(mediaDriver);
        //ensure reuse of aeron wherever possible
        if(aeron == null)
            aeron = Aeron.connect(getContext(mediaDriver));

        subscriber.setAeron(aeron);




        subscriber.run(args);
    }

    /**
     * Stop the server
     * @throws Exception
     */
    public void stop() throws Exception {
        if(server != null)
            server.stop();
        if(mediaDriver != null)
            CloseHelper.quietClose(mediaDriver);
        if(aeron != null)
            CloseHelper.quietClose(aeron);

    }



    private static  Aeron.Context getContext(MediaDriver mediaDriver) {
        return new Aeron.Context().publicationConnectionTimeout(-1)
                .availableImageHandler(AeronUtil::printAvailableImage)
                .unavailableImageHandler(AeronUtil::printUnavailableImage)
                .aeronDirectoryName(mediaDriver.aeronDirectoryName()).keepAliveInterval(1000)
                .errorHandler(e -> log.error(e.toString(), e));
    }


    public static void main(String[] args) {
        new ParameterServerNode().runMain(args);
    }

}
