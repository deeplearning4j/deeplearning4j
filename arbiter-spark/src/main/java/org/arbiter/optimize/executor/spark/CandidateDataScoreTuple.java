package org.arbiter.optimize.executor.spark;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.data.DataProvider;
import org.arbiter.optimize.api.score.ScoreFunction;

@AllArgsConstructor
@NoArgsConstructor
@Data
public class CandidateDataScoreTuple<C,D,M> {
    private  Candidate<C> candidate;
    private  DataProvider<D> dataProvider;
    private  ScoreFunction<M,D> scoreFunction;
}
