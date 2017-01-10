package org.nd4j.parameterserver.distributed.training;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.logic.Clipboard;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.co,
 */
public abstract class BaseTrainer<T extends TrainingMessage> implements TrainingDriver<T> {
    protected VoidConfiguration voidConfiguration;
    protected Transport transport;
    protected Clipboard clipboard;
    protected Storage storage;

    @Override
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Transport transport, @NonNull Storage storage, @NonNull Clipboard clipboard) {
        this.clipboard = clipboard;
        this.transport = transport;
        this.voidConfiguration = voidConfiguration;
        this.storage = storage;
    }

    protected int[] replicate(int value, int size) {
        int[] result = new int[size];
        for (int e = 0; e < size; e++)
            result[e] = value;

        return result;
    }
}
