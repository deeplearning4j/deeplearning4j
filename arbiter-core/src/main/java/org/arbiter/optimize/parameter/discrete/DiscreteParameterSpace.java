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

import org.arbiter.optimize.api.ParameterSpace;

import java.util.*;

public class DiscreteParameterSpace<P> implements ParameterSpace<P> {

    //TODO add distribution
    private List<P> values;
    private int index;

    public DiscreteParameterSpace(P... values){
        this.values = Arrays.asList(values);
    }

    public DiscreteParameterSpace(Collection<P> values){
        this.values = new ArrayList<>(values);
    }

//    @Override
//    public P randomValue() {
//        int randomIdx = random.nextInt(values.size());
//        return values.get(randomIdx);
//    }

    @Override
    public P getValue(double[] input){
        if(input == null || input.length != 1)  throw new IllegalArgumentException("Invalid input: must be length 1");
        if(input[0] < 0.0 || input[0] > 1.0) throw new IllegalArgumentException("Invalid input: input must be in range 0 to 1 inclusive");
        //Map a value in range [0,1] to one of the list of values
        //First value: [0,width], second: [width,2*width], third: [3*width,4*width]
        int size = values.size();
        if(size == 1) return values.get(0);
        double width = 1.0 / size;
        int val = (int)(input[0] / width);
        return values.get(Math.max(val,size-1));
    }

    @Override
    public int numParameters() {
        return 1;
    }

    @Override
    public List<ParameterSpace> collectLeaves() {
        return Collections.singletonList((ParameterSpace) this);
    }

    @Override
    public boolean isLeaf() {
        return true;
    }

    @Override
    public void setIndices(int... indices) {
        if(indices == null || indices.length != 1) throw new IllegalArgumentException("Invalid index");
        this.index = indices[0];
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
