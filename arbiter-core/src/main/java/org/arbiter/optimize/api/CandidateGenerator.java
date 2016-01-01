package org.arbiter.optimize.api;

public interface CandidateGenerator<T> {

    Candidate<T> getCandidate();

    void reportResults(Object result);  //TODO method signature

    ModelParameterSpace<T> getParameterSpace();

}
