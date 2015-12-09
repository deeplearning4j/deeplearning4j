package org.arbiter.optimize.api.saving;

import org.arbiter.optimize.api.OptimizationResult;

import java.io.IOException;

public interface ResultSaver<T,M> {

    void saveModel(OptimizationResult<T,M> result) throws IOException;

}
