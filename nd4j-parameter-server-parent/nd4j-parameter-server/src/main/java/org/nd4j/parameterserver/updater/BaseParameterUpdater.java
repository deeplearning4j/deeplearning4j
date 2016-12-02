package org.nd4j.parameterserver.updater;

import org.nd4j.parameterserver.updater.storage.InMemoryUpdateStorage;
import org.nd4j.parameterserver.updater.storage.UpdateStorage;

/**
 * Base class for the parameter updater
 * handling things such as update storage
 * and basic operations like reset and number of updates
 *
 * @author Adam Gibson
 */
public abstract class BaseParameterUpdater implements ParameterServerUpdater {
    protected UpdateStorage updateStorage;

    /**
     * Initialize this updater
     * with a custom update storage
     * @param updateStorage the update storage to use
     */
    public BaseParameterUpdater(UpdateStorage updateStorage) {
        this.updateStorage = updateStorage;
    }

    /**
     * Initializes this updater
     * with {@link InMemoryUpdateStorage}
     */
    public BaseParameterUpdater() {
        this(new InMemoryUpdateStorage());
    }



    /**
     * Reset internal counters
     * such as number of updates accumulated.
     */
    @Override
    public void reset() {
        updateStorage.clear();
    }


    /**
     * Num updates passed through
     * the updater
     *
     * @return the number of updates
     */
    @Override
    public int numUpdates() {
        return updateStorage.numUpdates();
    }
}
