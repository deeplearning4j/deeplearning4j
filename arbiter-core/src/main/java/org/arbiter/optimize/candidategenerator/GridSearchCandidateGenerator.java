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

import org.arbiter.optimize.api.Candidate;
import org.arbiter.optimize.api.ParameterSpace;
import org.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.arbiter.util.CollectionUtils;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;


public class GridSearchCandidateGenerator<T> extends BaseCandidateGenerator<T> {

    /** In what order should candidates be evaluated?<br>
     * Sequential:
     */
    public enum Mode {Sequential, Random};

    public static final int DEFAULT_DISCRETIZATION_COUNT = 10;

    private final int discretizationCount;
    private final Mode mode;

    private int[] numValuesPerParam;
    private Queue<Integer> order;

    private Random rng;

    public GridSearchCandidateGenerator(ParameterSpace<T> parameterSpace, Mode mode){
        this(parameterSpace, DEFAULT_DISCRETIZATION_COUNT, mode);
    }

    /**
     * @param parameterSpace
     * @param discretizationCount For continuous parameters: into how many values should we discretize them into?
     *                           For example, suppose continuous parameter is in range [0,1] with 3 bins:
     *                           do [0.0, 0.5, 1.0]
     */
    public GridSearchCandidateGenerator(ParameterSpace<T> parameterSpace, int discretizationCount, Mode mode){
        this(parameterSpace,discretizationCount,mode,new Random().nextLong());
    }
    public GridSearchCandidateGenerator(ParameterSpace<T> parameterSpace, int discretizationCount, Mode mode, long seed){
        super(parameterSpace);
        this.discretizationCount = discretizationCount;
        this.mode = mode;
        this.rng = new Random(seed);
    }

    @Override
    protected void initialize(){
        super.initialize();

        List<ParameterSpace> leaves = CollectionUtils.getUnique(parameterSpace.collectLeaves());
        int nParams = leaves.size();

        //Work out for each parameter: is it continuous or discrete?
        // for grid search: discrete values are grid-searchable as-is
        // continuous values: discretize using 'discretizationCount' bins
        // integer values: use min(max-min+1, discretizationCount) values. i.e., discretize if necessary
        numValuesPerParam = new int[nParams];
        long searchSize = 1;
        for( int i=0; i<nParams; i++ ){
            ParameterSpace ps = leaves.get(i);
            if(ps instanceof DiscreteParameterSpace){
                DiscreteParameterSpace dps = (DiscreteParameterSpace)ps;
                numValuesPerParam[i] = dps.numValues();
            } else if(ps instanceof IntegerParameterSpace){
                IntegerParameterSpace ips = (IntegerParameterSpace)ps;
                int min = ips.getMin();
                int max = ips.getMax();
                //Discretize, as some integer ranges are much too large to search (i.e., num. neural network units, between 100 and 1000)
                numValuesPerParam[i] = Math.min(max-min+1,discretizationCount);
            } else {
                numValuesPerParam[i] = discretizationCount;
            }
            searchSize *= numValuesPerParam[i];
            i++;
        }

        if(searchSize >= Integer.MAX_VALUE) throw new IllegalStateException("Invalid search: cannot process search with "
            + searchSize + " candidates > Integer.MAX_VALUE");  //TODO find a more reasonable upper bound?

        order = new ConcurrentLinkedQueue<>();

        int size = (int)searchSize;
        switch(mode){
            case Sequential:
                for( int i=0; i<size; i++ ){
                    order.add(i);
                }
                break;
            case Random:
                List<Integer> tempList = new ArrayList<>(size);
                for( int i=0; i<size; i++ ){
                    tempList.add(i);
                }
                Collections.shuffle(tempList, rng);
                order.addAll(tempList);
                break;
            default:
                throw new RuntimeException();
        }

    }

    @Override
    public Candidate<T> getCandidate() {
        int next = order.remove();

        //Next: max integer (candidate number) to values
        double[] values = indexToValues(next);

        return new Candidate<T>(parameterSpace.getValue(values),candidateCounter.getAndIncrement());
    }

    private double[] indexToValues(int candidateIdx){
        double[] out = new double[numValuesPerParam.length];

        //How? first map to index of num possible values. Then: to
        // 0-> [0,0,0], 1-> [1,0,0], 2-> [2,0,0], 3-> [0,1,0] etc



        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void reportResults(Object result) {

        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public String toString(){
        return "GridSearchCandidateGenerator(mode="+mode+")";
    }
}
