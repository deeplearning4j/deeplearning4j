package org.deeplearning4j.spark.parameterserver.networking.messages;

import org.nd4j.parameterserver.distributed.messages.BaseVoidMessage;

/**
 * @author raver119@gmail.com
 */
public class SilentIntroductoryConfirmation extends BaseVoidMessage {
    @Override
    public void processMessage() {
        /*
            we just want to get clearance before training starts here
         */
    }
}
