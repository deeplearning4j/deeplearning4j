package org.arbiter.optimize.randomsearch;

import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.CandidateGenerator;
import org.arbiter.optimize.api.ModelParameterSpace;

public class RandomSearchGenerator<T> implements CandidateGenerator<T> {

    private ModelParameterSpace<T> parameterSpace;

    public RandomSearchGenerator( ModelParameterSpace<T> parameterSpace ){
        this.parameterSpace = parameterSpace;
    }

    @Override
    public Candidate<T> getCandidate() {
        return new Candidate<T>(parameterSpace.randomCandidate());
    }
}
