package org.arbiter.optimize.api.score;

import org.arbiter.optimize.api.data.DataProvider;

import java.util.Map;

public interface ScoreFunction<M,D> {

    double score(M model, DataProvider<D> dataProvider, Map<String, Object> dataParameters);
}
