package org.deeplearning4j.spark.parameterserver.networking.messages;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;

/**
 * @author raver119@gmail.com
 */
public class SilentIntroductoryMessage extends BaseVoidMessage {
    protected String localIp;
    protected int port;

    protected SilentIntroductoryMessage() {
        //
    }

    public SilentIntroductoryMessage(@NonNull String localIP, int port) {
        this.localIp = localIP;
        this.port = port;
    }

    @Override
    public void processMessage() {
        /*
            basically we just want to send our IP, and get our new shardIndex in return. haha. bad idea obviously, but still...

            or, we can skip direct addressing here, use passive addressing instead, like in client mode?
         */

        transport.addShard(localIp, port);
        //transport.addClient(localIp, port);
    }
}
