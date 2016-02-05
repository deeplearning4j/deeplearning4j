/*
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */
package org.arbiter.optimize.candidategenerator;

import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.CandidateGenerator;
import org.arbiter.optimize.api.ParameterSpace;
import org.arbiter.util.CollectionUtils;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class RandomSearchGenerator<T> extends BaseCandidateGenerator<T> {



    public RandomSearchGenerator( ParameterSpace<T> parameterSpace ){
        super(parameterSpace);
    }



    @Override
    public Candidate<T> getCandidate() {

        double[] randomValues = Nd4j.rand(1,parameterSpace.numParameters()).data().asDouble();

        return new Candidate<T>(parameterSpace.getValue(randomValues),candidateCounter.getAndIncrement());
    }

    @Override
    public String toString(){
        return "RandomSearchCandidateGenerator()";
    }
}
