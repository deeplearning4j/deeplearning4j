package org.nd4j.parameterserver.updater;

import org.nd4j.aeron.ipc.NDArrayHolder;
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
    protected NDArrayHolder ndArrayHolder;

    public BaseParameterUpdater(UpdateStorage updateStorage, NDArrayHolder ndArrayHolder) {
        this.updateStorage = updateStorage;
        this.ndArrayHolder = ndArrayHolder;
    }

    /**
     * Returns true if the updater is
     * ready for a new array
     *
     * @return
     */
    @Override
    public boolean isReady() {
        return numUpdates() == requiredUpdatesForPass();
    }

    /**
     * Returns true if the
     * given updater is async
     * or synchronous
     * updates
     *
     * @return true if the given updater
     * is async or synchronous updates
     */
    @Override
    public boolean isAsync() {
        return true;
    }

    /**
     * Get the ndarray holder for this
     * updater
     *
     * @return the ndarray holder for this updater
     */
    @Override
    public NDArrayHolder ndArrayHolder() {
        return ndArrayHolder;
    }

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
