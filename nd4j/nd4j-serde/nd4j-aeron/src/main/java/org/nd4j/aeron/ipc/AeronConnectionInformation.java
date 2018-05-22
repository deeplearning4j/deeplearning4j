package org.nd4j.aeron.ipc;

import lombok.Builder;
import lombok.Data;

/**
 * Aeron connection information
 * pojo.
 * connectionHost represents the host for the media driver
 * connection host represents the port
 * stream represents the stream id to connect to
 * @author Adam Gibson
 */
@Data
@Builder
public class AeronConnectionInformation {
    private String connectionHost;
    private int connectionPort;
    private int streamId;

    /**
     * Traditional static generator method
     * @param connectionHost
     * @param connectionPort
     * @param streamId
     * @return
     */
    public static AeronConnectionInformation of(String connectionHost, int connectionPort, int streamId) {
        return AeronConnectionInformation.builder().connectionHost(connectionHost).connectionPort(connectionPort)
                        .streamId(streamId).build();
    }

    @Override
    public String toString() {
        return String.format("%s:%d:%d", connectionHost, connectionPort, streamId);
    }
}
