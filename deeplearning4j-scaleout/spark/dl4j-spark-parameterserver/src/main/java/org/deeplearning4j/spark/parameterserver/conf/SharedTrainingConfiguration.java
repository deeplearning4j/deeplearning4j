package org.deeplearning4j.spark.parameterserver.conf;

import lombok.*;
import org.deeplearning4j.optimize.solvers.accumulation.MessageHandler;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

import java.io.Serializable;

/**
 *
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class SharedTrainingConfiguration implements Serializable{
    protected VoidConfiguration voidConfiguration;

    // TODO: decide, if we abstract this one out, or not
    protected double threshold;
    protected String messageHandlerClass;


    public void setMessageHandlerClass(@NonNull String messageHandlerClass) {
        this.messageHandlerClass = messageHandlerClass;
    }

    public void setMessageHandlerClass(@NonNull MessageHandler handler) {
        this.messageHandlerClass = handler.getClass().getCanonicalName();
    }
}
