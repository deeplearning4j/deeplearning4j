package org.arbiter.optimize.api.score;

public interface ScoreFunction<M> {

    double score(M model);
}
