/*
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
 */

package org.arbiter.optimize.candidategenerator;

import org.arbiter.optimize.api.CandidateGenerator;
import org.arbiter.optimize.api.ParameterSpace;
import org.arbiter.util.CollectionUtils;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/** BaseCandidateGenerator: abstract class upon which {@link RandomSearchGenerator} and {@link GridSearchCandidateGenerator}
 * are built.
 * @param <T> Type of candidates to generate
 */
public abstract class BaseCandidateGenerator<T> implements CandidateGenerator<T> {

    protected ParameterSpace<T> parameterSpace;
    protected AtomicInteger candidateCounter = new AtomicInteger(0);

    public BaseCandidateGenerator(ParameterSpace<T> parameterSpace){
        this.parameterSpace = parameterSpace;
    }

    protected void initialize(){
        //First: collect leaf parameter spaces objects and remove duplicates
        List<ParameterSpace> noDuplicatesList = CollectionUtils.getUnique(parameterSpace.collectLeaves());

        //Second: assign each a number
        int i=0;
        for( ParameterSpace ps : noDuplicatesList){
            int np = ps.numParameters();
            if(np == 1){
                ps.setIndices(i++);
            } else {
                int[] values = new int[np];
                for( int j=0; j<np; j++ ) values[i++] = j;
                ps.setIndices(values);
            }
        }
    }

    @Override
    public ParameterSpace<T> getParameterSpace() {
        return parameterSpace;
    }

    @Override
    public void reportResults(Object result) {
        throw new UnsupportedOperationException("Not yet implemented");
    }
}
