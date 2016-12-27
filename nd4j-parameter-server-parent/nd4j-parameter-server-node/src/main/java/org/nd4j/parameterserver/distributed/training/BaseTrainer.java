package org.nd4j.parameterserver.distributed.training;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.conf.Configuration;
import org.nd4j.parameterserver.distributed.logic.Clipboard;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.co,
 */
public abstract class BaseTrainer<T extends TrainingMessage> implements TrainingDriver<T> {
    protected Configuration configuration;
    protected Transport transport;
    protected Clipboard clipboard;

    @Override
    public void init(@NonNull Configuration configuration, @NonNull Transport transport, @NonNull Clipboard clipboard) {
        this.clipboard = clipboard;
        this.transport = transport;
        this.configuration = configuration;
    }

    protected int[] replicate(int value, int size) {
        int[] result = new int[size];
        for (int e = 0; e < size; e++)
            result[e] = value;

        return result;
    }
}
