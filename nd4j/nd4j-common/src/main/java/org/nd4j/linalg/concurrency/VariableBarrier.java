package org.nd4j.linalg.concurrency;

public interface VariableBarrier {
    void registerConsumers(int numberOfConsumers);

    void synchronizedBlock();

    void desynchronizedBlock();

    void bypassEverything();

    void blockUntilSyncable();

    void blockUntilDesyncable();
}
