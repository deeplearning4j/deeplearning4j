package org.nd4j.parameterserver.updater;

import org.nd4j.parameterserver.updater.storage.InMemoryUpdateStorage;
import org.nd4j.parameterserver.updater.storage.UpdateStorage;

/**
 * Created by agibsonccc on 12/2/16.
 */
public abstract class BaseParameterUpdater implements ParameterServerUpdater {
    protected UpdateStorage updateStorage;

    public BaseParameterUpdater(UpdateStorage updateStorage) {
        this.updateStorage = updateStorage;
    }

    public BaseParameterUpdater() {
        this(new InMemoryUpdateStorage());
    }


}
