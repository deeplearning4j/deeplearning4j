package org.arbiter.optimize.executor.spark;

import lombok.AllArgsConstructor;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.score.ScoreFunction;

@AllArgsConstructor
public class CandidateDataScoreTuple<T,M,D> {
    private final Candidate<T> candidate;
    private final DataProvider<D> dataProvider;
    private final ScoreFunction<M,D> scoreFunction;
}
