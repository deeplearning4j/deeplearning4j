package org.arbiter.optimize.api;

import org.arbiter.optimize.api.data.DataProvider;

import java.util.concurrent.Callable;

public interface TaskCreator<T,M> {
    Callable<OptimizationResult<T,M>> create(Candidate<T> candidate, DataProvider<?> dataProvider);


}
