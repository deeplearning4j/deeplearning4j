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
    protected WorkspaceMode workspaceMode = WorkspaceMode.ENABLED;
    @Builder.Default
    protected int prefetchSize = 2;
    @Builder.Default
    protected boolean epochReset = false;
    @Builder.Default
    protected int numberOfWorkersPerNode = -1;
    @Builder.Default
    protected long debugLongerIterations = 0L;

    /**
     * This value **overrides** bufferSize calculations for gradients accumulator
     */
    @Builder.Default
    protected int bufferSize = 0;

    // TODO: decide, if we abstract this one out, or not
    @Builder.Default protected double threshold = 1e-3;
    @Builder.Default protected double thresholdStep = 1e-5;
    @Builder.Default protected double minThreshold = 1e-5;
    @Builder.Default protected double stepTrigger = 0.0;
    @Builder.Default protected int stepDelay = 3;
    @Builder.Default protected int shakeFrequency = 0;
    protected String messageHandlerClass;



    public void setMessageHandlerClass(@NonNull String messageHandlerClass) {
        this.messageHandlerClass = messageHandlerClass;
    }

    public void setMessageHandlerClass(@NonNull MessageHandler handler) {
        this.messageHandlerClass = handler.getClass().getCanonicalName();
    }
}
