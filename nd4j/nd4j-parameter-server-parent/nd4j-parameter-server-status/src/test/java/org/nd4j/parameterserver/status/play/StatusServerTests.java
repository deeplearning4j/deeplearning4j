package org.nd4j.parameterserver.status.play;

import org.junit.Test;
import play.server.Server;

/**
 * Created by agibsonccc on 12/1/16.
 */
public class StatusServerTests {

    @Test
    public void runStatusServer() {
        Server server = StatusServer.startServer(new InMemoryStatusStorage(), 65236);
        server.stop();
    }

}
