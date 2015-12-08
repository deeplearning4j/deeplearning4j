package org.arbiter.optimize.api;

public interface ParamSpace<T> {

    Candidate<T> randomCandidate();

}
