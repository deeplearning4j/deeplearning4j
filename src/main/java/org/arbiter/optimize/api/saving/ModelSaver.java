package org.arbiter.optimize.api.saving;

import java.io.IOException;

public interface ModelSaver<D> {

    void saveModel(D model) throws IOException;

}
