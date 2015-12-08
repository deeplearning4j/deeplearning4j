package org.arbiter.optimize.executor.spark;

import lombok.AllArgsConstructor;
import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.data.DataProvider;

@AllArgsConstructor
public class CandidateDataPair<T,D> {
    private final Candidate<T> candidate;
    private final DataProvider<D> dataProvider;
}
