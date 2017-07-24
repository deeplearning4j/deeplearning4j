package org.deeplearning4j.spark.parameterserver.conf;

import lombok.*;
import org.deeplearning4j.nn.conf.WorkspaceMode;
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
public class SharedTrainingConfiguration implements Serializable {
    protected VoidConfiguration voidConfiguration;

    @Builder.Default
    protected WorkspaceMode workspaceMode = WorkspaceMode.SEPARATE;
    @Builder.Default
    protected int prefetchSize = 2;
    @Builder.Default
    protected boolean epochReset = false;
    @Builder.Default
    protected int numberOfWorkersPerNode = -1;
    @Builder.Default
    protected long debugLongerIterations = 0L;

    // TODO: decide, if we abstract this one out, or not
    protected double threshold;
    protected double thresholdStep = 1e-5;
    protected double minThreshold = 1e-5;
    protected double stepTrigger;
    protected int stepDelay;
    protected int shakeFrequency = 0;
    protected String messageHandlerClass;



    public void setMessageHandlerClass(@NonNull String messageHandlerClass) {
        this.messageHandlerClass = messageHandlerClass;
    }

    public void setMessageHandlerClass(@NonNull MessageHandler handler) {
        this.messageHandlerClass = handler.getClass().getCanonicalName();
    }
}
