package org.deeplearning4j.spark.parameterserver.networking.messages;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;
import org.nd4j.parameterserver.distributed.messages.DistributedMessage;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SilentIntroductoryMessage extends BaseVoidMessage implements DistributedMessage {
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

        log.info("Adding client {}:{}", localIp, port);
        //transport.addShard(localIp, port);
        transport.addClient(localIp, port);
    }

    @Override
    public boolean isBlockingMessage() {
        // this is blocking message, we want to get reply back before going further
        return true;
    }
}
