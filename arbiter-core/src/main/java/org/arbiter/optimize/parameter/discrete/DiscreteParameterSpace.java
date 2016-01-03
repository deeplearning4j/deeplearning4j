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
package org.arbiter.optimize.parameter.discrete;

import org.arbiter.optimize.parameter.ParameterSpace;

import java.util.*;

public class DiscreteParameterSpace<P> implements ParameterSpace<P> {

    //TODO add distribution
    private List<P> values;
    private Random random = new Random();;  //TODO handling of seeds in global config

    public DiscreteParameterSpace(P... values){
        this.values = Arrays.asList(values);
    }

    public DiscreteParameterSpace(Collection<P> values){
        this.values = new ArrayList<>(values);
    }

    @Override
    public P randomValue() {
        int randomIdx = random.nextInt(values.size());
        return values.get(randomIdx);
    }

    @Override
    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append("DiscreteParameterSpace(");
        int n = values.size();
        for( int i=0; i<n; i++ ){
            sb.append(values.get(i));
            sb.append( (i == n-1 ? ")" : ","));
        }
        return sb.toString();
    }
}
