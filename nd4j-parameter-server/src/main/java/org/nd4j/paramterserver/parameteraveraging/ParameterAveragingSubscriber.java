package org.nd4j.paramterserver.parameteraveraging;

import io.aeron.Aeron;
import org.nd4j.aeron.ipc.AeronNDArraySubscriber;
import org.nd4j.aeron.ipc.AeronUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Subscriber main class for the parameter
 * averaging server
 *
 * @author Adam Gibson
 */
public class ParameterAveragingSubscriber {
    private static Logger log = LoggerFactory.getLogger(ParameterAveragingSubscriber.class);

    private ParameterAveragingSubscriber() {}

    public static void main(String[] args) {
        Aeron.Context ctx = new Aeron.Context().publicationConnectionTimeout(-1).availableImageHandler(AeronUtil::printAvailableImage)
                .unavailableImageHandler(AeronUtil::printUnavailableImage)
                .aeronDirectoryName(args[0]).keepAliveInterval(1000)
                .errorHandler(e -> log.error(e.toString(), e));
        int length = Integer.parseInt(args[1]);
        ParameterAveragingListener subscriber = new ParameterAveragingListener(length);
        AeronNDArraySubscriber subscriber1 =  AeronNDArraySubscriber.startSubscriber(ctx,"localhost",2800,subscriber,10);

    }
}
