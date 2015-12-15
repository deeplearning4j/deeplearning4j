package org.arbiter.optimize.randomsearch;

import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.CandidateGenerator;
import org.arbiter.optimize.api.ModelParameterSpace;

import java.util.concurrent.atomic.AtomicInteger;

public class RandomSearchGenerator<T> implements CandidateGenerator<T> {

    private ModelParameterSpace<T> parameterSpace;
    private AtomicInteger candidateCounter = new AtomicInteger(0);

    public RandomSearchGenerator( ModelParameterSpace<T> parameterSpace ){
        this.parameterSpace = parameterSpace;
    }

    @Override
    public Candidate<T> getCandidate() {
        return new Candidate<T>(parameterSpace.randomCandidate(),candidateCounter.getAndIncrement());
    }

    @Override
    public void reportResults(Object result) {
        throw new UnsupportedOperationException("Not yet implemented");
    }
}
