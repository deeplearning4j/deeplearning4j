package org.nd4j.parameterserver.node;

import io.aeron.driver.MediaDriver;
import lombok.extern.slf4j.Slf4j;
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
public class ParameterServerNode {
    private Server server;
    private ParameterServerSubscriber subscriber;
    private MediaDriver mediaDriver;

    public void runMain(String[] args) {
        server = StatusServer.startServer(new InMemoryStatusStorage(),9000);
        mediaDriver = MediaDriver.launchEmbedded();
        log.info("Started media driver with aeron directory " + mediaDriver.aeronDirectoryName());
        subscriber = new ParameterServerSubscriber(mediaDriver);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if(server != null)
                server.stop();
            if(mediaDriver != null)
                mediaDriver.close();

        }));
        subscriber.run(args);
    }

    public static void main(String[] args) {
        new ParameterServerNode().runMain(args);
    }

}
