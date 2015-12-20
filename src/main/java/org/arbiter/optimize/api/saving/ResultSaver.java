package org.arbiter.optimize.api.saving;

import org.arbiter.optimize.api.OptimizationResult;

import java.io.IOException;

public interface ResultSaver<T,M,A> {

    ResultReference<T,M,A> saveModel(OptimizationResult<T,M,A> result) throws IOException;

}
