/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.arbiter.optimize.generator;

import lombok.EqualsAndHashCode;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.random.RandomAdaptor;
import org.deeplearning4j.arbiter.optimize.api.Candidate;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.discrete.DiscreteParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.util.LeafUtils;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;


/**
 * GridSearchCandidateGenerator: generates candidates in an exhaustive grid search manner.<br>
 * Note that:<br>
 * - For discrete parameters: the grid size (# values to check per hyperparameter) is equal to the number of values for
 * that hyperparameter<br>
 * - For integer parameters: the grid size is equal to {@code min(discretizationCount,max-min+1)}. Some integer ranges can
 * be large, and we don't necessarily want to exhaustively search them. {@code discretizationCount} is a constructor argument<br>
 * - For continuous parameters: the grid size is equal to {@code discretizationCount}.<br>
 * In all cases, the minimum, maximum and gridSize-2 values between the min/max will be generated.<br>
 * Also note that: if a probability distribution is provided for continuous hyperparameters, this will be taken into account
 * when generating candidates. This allows the grid for a hyperparameter to be non-linear: i.e., for example, linear in log space
 *
 * @author Alex Black
 */
@Slf4j
@EqualsAndHashCode(exclude = {"order"}, callSuper = true)
@JsonIgnoreProperties({"numValuesPerParam", "totalNumCandidates", "order", "candidateCounter", "rng", "candidate"})
public class GridSearchCandidateGenerator extends BaseCandidateGenerator {

    /**
     * In what order should candidates be generated?<br>
     * <b>Sequential</b>: generate candidates in order. The first hyperparameter will be changed most rapidly, and the last
     * will be changed least rapidly.<br>
     * <b>RandomOrder</b>: generate candidates in a random order<br>
     * In both cases, the same candidates will be generated; only the order of generation is different
     */
    public enum Mode {
        Sequential, RandomOrder
    }

    private final int discretizationCount;
    private final Mode mode;

    private int[] numValuesPerParam;
    private int totalNumCandidates;
    private Queue<Integer> order;

    /**
     * @param parameterSpace      ParameterSpace from which to generate candidates
     * @param discretizationCount For continuous parameters: into how many values should we discretize them into?
     *                            For example, suppose continuous parameter is in range [0,1] with 3 bins:
     *                            do [0.0, 0.5, 1.0]. Note that if all values
     * @param mode                {@link GridSearchCandidateGenerator.Mode} specifies the order
     *                            in which candidates should be generated.
     */
    public GridSearchCandidateGenerator(@JsonProperty("parameterSpace") ParameterSpace<?> parameterSpace,
                    @JsonProperty("discretizationCount") int discretizationCount, @JsonProperty("mode") Mode mode,
                    @JsonProperty("dataParameters") Map<String, Object> dataParameters,
                    @JsonProperty("initDone") boolean initDone) {
        super(parameterSpace, dataParameters, initDone);
        this.discretizationCount = discretizationCount;
        this.mode = mode;
        initialize();
    }

    /**
     * @param parameterSpace      ParameterSpace from which to generate candidates
     * @param discretizationCount For continuous parameters: into how many values should we discretize them into?
     *                            For example, suppose continuous parameter is in range [0,1] with 3 bins:
     *                            do [0.0, 0.5, 1.0]. Note that if all values
     * @param mode                {@link GridSearchCandidateGenerator.Mode} specifies the order
     *                            in which candidates should be generated.
     */
    public GridSearchCandidateGenerator(ParameterSpace<?> parameterSpace, int discretizationCount, Mode mode,
                                        Map<String, Object> dataParameters){
        this(parameterSpace, discretizationCount, mode, dataParameters, false);
    }

    @Override
    protected void initialize() {
        super.initialize();

        List<ParameterSpace> leaves = LeafUtils.getUniqueObjects(parameterSpace.collectLeaves());
        int nParams = leaves.size();

        //Work out for each parameter: is it continuous or discrete?
        // for grid search: discrete values are grid-searchable as-is
        // continuous values: discretize using 'discretizationCount' bins
        // integer values: use min(max-min+1, discretizationCount) values. i.e., discretize if necessary
        numValuesPerParam = new int[nParams];
        long searchSize = 1;
        for (int i = 0; i < nParams; i++) {
            ParameterSpace ps = leaves.get(i);
            if (ps instanceof DiscreteParameterSpace) {
                DiscreteParameterSpace dps = (DiscreteParameterSpace) ps;
                numValuesPerParam[i] = dps.numValues();
            } else if (ps instanceof IntegerParameterSpace) {
                IntegerParameterSpace ips = (IntegerParameterSpace) ps;
                int min = ips.getMin();
                int max = ips.getMax();
                //Discretize, as some integer ranges are much too large to search (i.e., num. neural network units, between 100 and 1000)
                numValuesPerParam[i] = Math.min(max - min + 1, discretizationCount);
            } else {
                numValuesPerParam[i] = discretizationCount;
            }
            searchSize *= numValuesPerParam[i];
        }

        if (searchSize >= Integer.MAX_VALUE)
            throw new IllegalStateException("Invalid search: cannot process search with " + searchSize
                            + " candidates > Integer.MAX_VALUE"); //TODO find a more reasonable upper bound?

        order = new ConcurrentLinkedQueue<>();

        totalNumCandidates = (int) searchSize;
        switch (mode) {
            case Sequential:
                for (int i = 0; i < totalNumCandidates; i++) {
                    order.add(i);
                }
                break;
            case RandomOrder:
                List<Integer> tempList = new ArrayList<>(totalNumCandidates);
                for (int i = 0; i < totalNumCandidates; i++) {
                    tempList.add(i);
                }

                Collections.shuffle(tempList, new RandomAdaptor(rng));
                order.addAll(tempList);
                break;
            default:
                throw new RuntimeException();
        }

    }

    @Override
    public boolean hasMoreCandidates() {
        return !order.isEmpty();
    }

    @Override
    public Candidate getCandidate() {
        int next = order.remove();

        //Next: max integer (candidate number) to values
        double[] values = indexToValues(numValuesPerParam, next, totalNumCandidates);

        Object value = null;
        Exception e = null;
        try {
            value = parameterSpace.getValue(values);
        } catch (Exception e2) {
            log.warn("Error getting configuration for candidate", e2);
            e = e2;
        }

        return new Candidate(value, candidateCounter.getAndIncrement(), values, dataParameters, e);
    }

    @Override
    public Class<?> getCandidateType() {
        return null;
    }

    public static double[] indexToValues(int[] numValuesPerParam, int candidateIdx, int product) {
        //How? first map to index of num possible values. Then: to double values in range 0 to 1
        // 0-> [0,0,0], 1-> [1,0,0], 2-> [2,0,0], 3-> [0,1,0] etc
        //Based on: Nd4j Shape.ind2sub

        int denom = product;
        int num = candidateIdx;
        int[] index = new int[numValuesPerParam.length];

        for (int i = index.length - 1; i >= 0; i--) {
            denom /= numValuesPerParam[i];
            index[i] = num / denom;
            num %= denom;
        }

        //Now: convert indexes to values in range [0,1]
        //min value -> 0
        //max value -> 1
        double[] out = new double[numValuesPerParam.length];
        for (int i = 0; i < out.length; i++) {
            if (numValuesPerParam[i] <= 1)
                out[i] = 0.0;
            else {
                out[i] = index[i] / ((double) (numValuesPerParam[i] - 1));
            }
        }

        return out;
    }

    @Override
    public String toString() {
        return "GridSearchCandidateGenerator(mode=" + mode + ")";
    }
}
