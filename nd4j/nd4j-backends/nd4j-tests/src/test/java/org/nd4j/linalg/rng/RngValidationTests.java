/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.linalg.rng;

import lombok.Builder;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.OpValidationSuite;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.bool.MatchConditionTransform;
import org.nd4j.linalg.api.ops.random.compat.RandomStandardNormal;
import org.nd4j.linalg.api.ops.random.custom.DistributionUniform;
import org.nd4j.linalg.api.ops.random.custom.RandomBernoulli;
import org.nd4j.linalg.api.ops.random.custom.RandomExponential;
import org.nd4j.linalg.api.ops.random.impl.*;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

@Slf4j
public class RngValidationTests extends BaseNd4jTest {

    public RngValidationTests(Nd4jBackend b){
        super(b);
    }

    @Override
    public char ordering(){
        return 'c';
    }

    @Builder(builderClassName = "TestCaseBuilder")
    @Data
    public static class TestCase {
        private String opType;
        private DataType dataType;
        @Builder.Default private long rngSeed = 12345;
        private long[] shape;
        private double minValue;
        private double maxValue;
        private boolean minValueInclusive;
        private boolean maxValueInclusive;
        private Double expectedMean;
        private Double expectedStd;
        @Builder.Default private double meanRelativeErrorTolerance = 0.01;
        @Builder.Default private double stdRelativeErrorTolerance = 0.01;
        private Double meanMinAbsErrorTolerance;    //Consider relative error between 0 and 0.001: relative error is 1.0, but absolute error is small
        private Double stdMinAbsErrorTolerance;
        @Builder.Default private Map<String,Object> args = new LinkedHashMap<>();

        public static class TestCaseBuilder {

            public TestCaseBuilder arg(String arg, Object value){
                if(args == null) {
                    args(new LinkedHashMap<>());
                }
                args.put(arg, value);
                return this;
            }

            public TestCaseBuilder shape(long... shape){
                this.shape = shape;
                return this;
            }
        }

        public INDArray arr(){
            Preconditions.checkState(shape != null, "Shape is null");
            INDArray arr = Nd4j.createUninitialized(dataType, shape);
            arr.assign(Double.NaN);     //Assign NaNs to help detect implementation issues
            return arr;
        }

        public <T> T prop(String s){
            Preconditions.checkState(args != null && args.containsKey(s), "Property \"%s\" not found. All properties: %s", s, args);
            return (T)args.get(s);
        }
    }


    @Test
    public void validateRngDistributions(){
        OpValidationSuite.ignoreFailing();      //https://github.com/deeplearning4j/deeplearning4j/issues/6958 - 2018-01-09

        List<TestCase> testCases = new ArrayList<>();
        for(DataType type : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}) {
            //Legacy (non-custom) RNG ops:
            testCases.add(TestCase.builder().opType("bernoulli").dataType(type).shape(new long[0]).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("prob", 0.5).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("bernoulli").dataType(type).shape(1000).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("prob", 0.5)
                    .expectedMean(0.5).expectedStd(Math.sqrt(0.5*0.5) /*var = p*(1-p)*/).build());
            testCases.add(TestCase.builder().opType("bernoulli").dataType(type).shape(100,10000).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("prob", 0.2)
                    .expectedMean(0.2).expectedStd(Math.sqrt(0.2*(1-0.2)) /*var = p*(1-p)*/).meanRelativeErrorTolerance(0.005).stdRelativeErrorTolerance(0.01).build());

            testCases.add(TestCase.builder().opType("uniform").dataType(type).shape(new long[0]).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("min", 0.0).arg("max", 1.0).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("uniform").dataType(type).shape(1000).minValue(1).maxValue(2).minValueInclusive(true).maxValueInclusive(true).arg("min", 1.0).arg("max",2.0)
                    .expectedMean((1+2)/2.0).expectedStd(Math.sqrt(1/12.0 * Math.pow(2.0-1.0, 2)) /*Var: 1/12 * (b-a)^2*/).build());
            testCases.add(TestCase.builder().opType("uniform").dataType(type).shape(100,10000).minValue(-4).maxValue(-2).minValueInclusive(true).maxValueInclusive(true).arg("min", -4.0).arg("max",-2.0)
                    .expectedMean(-3.0).expectedStd(Math.sqrt(1/12.0 * Math.pow(-4.0+2.0, 2)) /*Var: 1/12 * (b-a)^2*/).meanRelativeErrorTolerance(0.005).stdRelativeErrorTolerance(0.01).build());

            testCases.add(TestCase.builder().opType("gaussian").dataType(type).shape(new long[0]).minValue(minValue(type)).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true).arg("mean", 0.0).arg("std", 1.0).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("gaussian").dataType(type).shape(1000).minValue(minValue(type)).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true).arg("mean", 0.0).arg("std", 1.0)
                    .expectedMean(0.0).expectedStd(1.0).stdRelativeErrorTolerance(0.03).meanMinAbsErrorTolerance(0.1).stdMinAbsErrorTolerance(0.1).build());
            testCases.add(TestCase.builder().opType("gaussian").dataType(type).shape(100,1000).minValue(minValue(type)).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true).arg("mean", 2.0).arg("std", 0.5)
                    .expectedMean(2.0).expectedStd(0.5).meanRelativeErrorTolerance(0.01).stdRelativeErrorTolerance(0.01).meanMinAbsErrorTolerance(0.001).build());

            testCases.add(TestCase.builder().opType("binomial").dataType(type).shape(new long[0]).minValue(0).maxValue(5).minValueInclusive(true).maxValueInclusive(true).arg("n", 5).arg("p",0.5).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("binomial").dataType(type).shape(1000).minValue(0).maxValue(10).minValueInclusive(true).maxValueInclusive(true).arg("n", 10).arg("p",0.5)
                    .stdRelativeErrorTolerance(0.02).expectedMean(10*0.5).expectedStd(Math.sqrt(10*0.5*(1-0.5)) /*var = np(1-p)*/).build());
            testCases.add(TestCase.builder().opType("binomial").dataType(type).shape(100,10000).minValue(0).maxValue(20).minValueInclusive(true).maxValueInclusive(true).arg("n", 20).arg("p",0.2)
                    .expectedMean(20*0.2).expectedStd(Math.sqrt(20*0.2*(1-0.2)) /*var = np(1-p)*/).meanRelativeErrorTolerance(0.001).stdRelativeErrorTolerance(0.01).build());

                //truncated normal clips at (mean-2*std, mean+2*std). Mean for equal 2 sided clipping about mean is same as original mean. Variance is difficult to calculate...
                //Assume variance is similar to non-truncated normal (should be a bit less in practice) but use large relative error here
            testCases.add(TestCase.builder().opType("truncated_normal").dataType(type).shape(new long[0]).minValue(-2.0).maxValue(2.0).minValueInclusive(true).maxValueInclusive(true).arg("mean", 0.0).arg("std", 1.0).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("truncated_normal").dataType(type).shape(1000).minValue(-2.0).maxValue(2.0).minValueInclusive(true).maxValueInclusive(true).arg("mean", 0.0).arg("std", 1.0)
                    .expectedMean(0.0).expectedStd(1.0).stdRelativeErrorTolerance(0.2).meanMinAbsErrorTolerance(0.1).build());
            testCases.add(TestCase.builder().opType("truncated_normal").dataType(type).shape(100,10000).minValue(1.0).maxValue(3.0).minValueInclusive(true).maxValueInclusive(true).arg("mean", 2.0).arg("std", 0.5)
                    .expectedMean(2.0).expectedStd(0.5).meanRelativeErrorTolerance(0.001).stdRelativeErrorTolerance(0.2).meanMinAbsErrorTolerance(0.001).build());

            //Dropout (non-inverted): same as bernoulli distribution, when dropout applied to "ones" array
            testCases.add(TestCase.builder().opType("dropout").dataType(type).shape(new long[0]).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("p", 0.5).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("dropout").dataType(type).shape(1000).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("p", 0.4)
                    .expectedMean(0.4).expectedStd(Math.sqrt(0.4*(1-0.4)) /*var = p*(1-p)*/).meanMinAbsErrorTolerance(0.05).stdMinAbsErrorTolerance(0.05).build());
            testCases.add(TestCase.builder().opType("dropout").dataType(type).shape(100,10000).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("p", 0.3)
                    .expectedMean(0.3).expectedStd(Math.sqrt(0.3*(1-0.3)) /*var = p*(1-p)*/).meanRelativeErrorTolerance(0.005).stdRelativeErrorTolerance(0.01).build());

            //Dropout (inverted): basically bernoulli distribution * 2, when inverted dropout applied to "ones" array
            testCases.add(TestCase.builder().opType("dropout_inverted").dataType(type).shape(new long[0]).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("p", 0.5).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("dropout_inverted").dataType(type).shape(1000).minValue(0).maxValue(1.0/0.4).minValueInclusive(true).maxValueInclusive(true).arg("p", 0.4)
                    //Mean: 0.4 probability of  being retained - mean is 0.4 probability * (1.0/0.4) = 1.0. i.e., expected mean is unchanged by inverted dropout
                    .expectedMean(1.0).expectedStd(1/0.4*Math.sqrt(0.4*(1-0.4)) /*var = p*(1-p)*/).meanMinAbsErrorTolerance(0.05).stdMinAbsErrorTolerance(0.05).build());
            testCases.add(TestCase.builder().opType("dropout_inverted").dataType(type).shape(100,10000).minValue(0).maxValue(1.0/0.3).minValueInclusive(true).maxValueInclusive(true).arg("p", 0.3)
                    .expectedMean(1.0).expectedStd(1/0.3*Math.sqrt(0.3*(1-0.3)) /*var = p*(1-p); note var(aX) = a^2 var(X)*/).meanRelativeErrorTolerance(0.005).stdRelativeErrorTolerance(0.01).build());

            //Linspace: we'll treat is as basically a uniform distribution for the purposes of these tests...
            testCases.add(TestCase.builder().opType("linspace").dataType(type).shape(1000).minValue(1).maxValue(2).minValueInclusive(true).maxValueInclusive(true).arg("from", 1.0).arg("to",2.0)
                    .expectedMean(1.5).expectedStd(Math.sqrt(1/12.0 * Math.pow(2.0-1.0, 2)) /*Var: 1/12 * (b-a)^2*/).build());

            //Log normal distribution: parameterized such that if X~lognormal(m,s) then mean(log(X))=m and std(log(X))=s
            //mean is given by exp(mu+s^2/2), variance [exp(s^2)-1]*[exp(2*mu+s^2)]
            testCases.add(TestCase.builder().opType("lognormal").dataType(type).shape(new long[0]).minValue(0).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true)
                    .arg("mu", 0.0).arg("s", 1.0).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("lognormal").dataType(type).shape(1000).minValue(0).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true)
                    .arg("mu", 0.0).arg("s", 1.0).expectedMean(Math.exp(0.0 + 1.0/2.0)).expectedStd(Math.sqrt((Math.exp(1.0)-1)*Math.exp(1.0)) ).meanRelativeErrorTolerance(0.1).stdRelativeErrorTolerance(0.1)
                    .meanMinAbsErrorTolerance(0.1).stdMinAbsErrorTolerance(0.1).build());
            testCases.add(TestCase.builder().opType("lognormal").dataType(type).shape(100,10000).minValue(0).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true).arg("mu", 2.0).arg("s", 0.5)
                    .expectedMean(Math.exp(2.0 + 0.5*0.5/2.0)).expectedStd(Math.sqrt((Math.exp(0.5*0.5)-1)*Math.exp(2.0*2.0+0.5*0.5))).meanRelativeErrorTolerance(0.01).stdRelativeErrorTolerance(0.01).meanMinAbsErrorTolerance(0.001).build());

            //Choice op. For the purposes of this test, use discrete uniform distribution with values 0 to 10 inclusive
            testCases.add(TestCase.builder().opType("choice").dataType(type).shape(new long[0]).minValue(0).maxValue(10).minValueInclusive(true).maxValueInclusive(true).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("choice").dataType(type).shape(1000).minValue(0).maxValue(10).minValueInclusive(true).maxValueInclusive(true)
                    .expectedMean(5.0 /*(a+b)/2 */).expectedStd(Math.sqrt((Math.pow(10-0+1,2)-1)/12.0) /* variance = ((b-a+1)^2-1)/12 */).meanRelativeErrorTolerance(0.05).stdRelativeErrorTolerance(0.05)
                    .meanMinAbsErrorTolerance(0.05).stdMinAbsErrorTolerance(0.05).build());
            testCases.add(TestCase.builder().opType("choice").dataType(type).shape(100,10000).minValue(0).maxValue(10).minValueInclusive(true).maxValueInclusive(true)
                    .expectedMean(5.0 /*(a+b)/2 */).expectedStd(Math.sqrt((Math.pow(10-0+1,2)-1)/12.0) /* variance = ((b-a+1)^2-1)/12 */).meanRelativeErrorTolerance(0.01).stdRelativeErrorTolerance(0.01).meanMinAbsErrorTolerance(0.001).build());

            //Probabilistic merge: use 0 and 1, 0.5 probability. Then it's same as bernoulli distribution
            testCases.add(TestCase.builder().opType("probabilisticmerge").dataType(type).shape(new long[0]).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("prob", 0.5).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("probabilisticmerge").dataType(type).shape(1000).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("prob", 0.5)
                    .expectedMean(0.5).expectedStd(Math.sqrt(0.5*0.5) /*var = p*(1-p)*/).build());
            testCases.add(TestCase.builder().opType("probabilisticmerge").dataType(type).shape(100,10000).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("prob", 0.2)
                    .expectedMean(0.2).expectedStd(Math.sqrt(0.2*(1-0.2)) /*var = p*(1-p)*/).meanRelativeErrorTolerance(0.005).stdRelativeErrorTolerance(0.01).build());

            //Range: x to y in N steps - essentially same statistical properties as uniform distribution
            testCases.add(TestCase.builder().opType("range").dataType(type).shape(10).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("min", 0.0).arg("max", 1.0).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("range").dataType(type).shape(1000).minValue(1).maxValue(2).minValueInclusive(true).maxValueInclusive(true).arg("min", 1.0).arg("max",2.0)
                    .expectedMean((1+2)/2.0).expectedStd(Math.sqrt(1/12.0 * Math.pow(2.0-1.0, 2)) /*Var: 1/12 * (b-a)^2*/).build());

            //AlphaDropout: implements a * (x * d + alphaPrime * (1-d)) + b, where d ~ Bernoulli(p), i.e., d \in {0,1}.
            //For ones input and p=0.5, this should give us values (a+b or a*alphaPrime+b) with probability 0.5
            //Mean should be same as input - i.e., 1
            testCases.add(TestCase.builder().opType("alphaDropout").dataType(type).shape(new long[0]).maxValue(alphaDropoutA(0.5)+alphaDropoutB(0.5))
                    .minValue(alphaDropoutA(0.5)*ALPHA_PRIME+alphaDropoutB(0.5)).minValueInclusive(true).maxValueInclusive(true).arg("p", 0.5).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("alphaDropout").dataType(type).shape(1000).maxValue(alphaDropoutA(0.4)+alphaDropoutB(0.4))
                    .minValue(alphaDropoutA(0.4)*ALPHA_PRIME+alphaDropoutB(0.4)).minValueInclusive(true).maxValueInclusive(true).arg("p", 0.4)
                    //Mean: 0.4 probability of  being retained - mean is 0.4 probability * (1.0/0.4) = 1.0. i.e., expected mean is unchanged by inverted dropout
                    .expectedMean(1.0).build());
            testCases.add(TestCase.builder().opType("alphaDropout").dataType(type).shape(100,10000).maxValue(alphaDropoutA(0.3)+alphaDropoutB(0.3))
                    .minValue(alphaDropoutA(0.3)*ALPHA_PRIME+alphaDropoutB(0.3)).minValueInclusive(true).maxValueInclusive(true).arg("p", 0.3)
                    .expectedMean(1.0).meanRelativeErrorTolerance(0.005).stdRelativeErrorTolerance(0.01).build());


            //--- Custom ops ---
            //DistributionUniform, RandomBernoulli, RandomExponential, RandomNormal, RandomStandardNormal
            testCases.add(TestCase.builder().opType("distributionuniform").dataType(type).shape(new long[0]).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("min", 0.0).arg("max", 1.0).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("distributionuniform").dataType(type).shape(1000).minValue(1).maxValue(2).minValueInclusive(true).maxValueInclusive(true).arg("min", 1.0).arg("max",2.0)
                    .expectedMean((1+2)/2.0).expectedStd(Math.sqrt(1/12.0 * Math.pow(2.0-1.0, 2)) /*Var: 1/12 * (b-a)^2*/).build());
            testCases.add(TestCase.builder().opType("distributionuniform").dataType(type).shape(100,10000).minValue(-4).maxValue(-2).minValueInclusive(true).maxValueInclusive(true).arg("min", -4.0).arg("max",-2.0)
                    .expectedMean(-3.0).expectedStd(Math.sqrt(1/12.0 * Math.pow(-4.0+2.0, 2)) /*Var: 1/12 * (b-a)^2*/).meanRelativeErrorTolerance(0.005).stdRelativeErrorTolerance(0.01).build());

            testCases.add(TestCase.builder().opType("randombernoulli").dataType(type).shape(new long[0]).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("prob", 0.5).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("randombernoulli").dataType(type).shape(1000).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("prob", 0.5)
                    .expectedMean(0.5).expectedStd(Math.sqrt(0.5*0.5) /*var = p*(1-p)*/).build());
            testCases.add(TestCase.builder().opType("randombernoulli").dataType(type).shape(100,10000).minValue(0).maxValue(1).minValueInclusive(true).maxValueInclusive(true).arg("prob", 0.2)
                    .expectedMean(0.2).expectedStd(Math.sqrt(0.2*(1-0.2)) /*var = p*(1-p)*/).meanRelativeErrorTolerance(0.005).stdRelativeErrorTolerance(0.01).build());

            //3 cases: lambda = 1, 1, 0.4
            testCases.add(TestCase.builder().opType("randomexponential").dataType(type).shape(new long[0]).minValue(0).maxValue(maxValue(type)).minValueInclusive(false).maxValueInclusive(true).arg("lambda", 1.0).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("randomexponential").dataType(type).shape(1000).minValue(0.0).maxValue(maxValue(type)).minValueInclusive(false).maxValueInclusive(true).arg("lambda", 1.0)
                    .expectedMean(1.0).expectedStd(1.0 /*var = 1 / lambda^2*/).build());
            testCases.add(TestCase.builder().opType("randomexponential").dataType(type).shape(100,10000).minValue(0.0).maxValue(maxValue(type)).minValueInclusive(false).maxValueInclusive(true).arg("lambda", 0.4)
                    .expectedMean(1.0 / 0.4).expectedStd(1.0 / Math.pow(0.4, 2) /*var = 1 / lambda^2*/).meanRelativeErrorTolerance(0.005).stdRelativeErrorTolerance(0.01).build());

            testCases.add(TestCase.builder().opType("randomnormal").dataType(type).shape(new long[0]).minValue(minValue(type)).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true).arg("mean", 0.0).arg("std", 1.0).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("randomnormal").dataType(type).shape(1000).minValue(minValue(type)).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true).arg("mean", 0.0).arg("std", 1.0)
                    .expectedMean(0.0).expectedStd(1.0).meanMinAbsErrorTolerance(0.05).stdMinAbsErrorTolerance(0.05).build());
            testCases.add(TestCase.builder().opType("randomnormal").dataType(type).shape(100,1000).minValue(minValue(type)).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true).arg("mean", 2.0).arg("std", 0.5)
                    .expectedMean(2.0).expectedStd(0.5).meanRelativeErrorTolerance(0.01).stdRelativeErrorTolerance(0.01).meanMinAbsErrorTolerance(0.001).build());

            testCases.add(TestCase.builder().opType("randomstandardnormal").dataType(type).shape(new long[0]).minValue(minValue(type)).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true).build());       //Don't check mean/std for 1 element
            testCases.add(TestCase.builder().opType("randomstandardnormal").dataType(type).shape(1000).minValue(minValue(type)).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true)
                    .expectedMean(0.0).expectedStd(1.0).meanMinAbsErrorTolerance(0.05).stdMinAbsErrorTolerance(0.05).build());
            testCases.add(TestCase.builder().opType("randomstandardnormal").dataType(type).shape(100,1000).minValue(minValue(type)).maxValue(maxValue(type)).minValueInclusive(true).maxValueInclusive(true)
                    .expectedMean(0.0).expectedStd(1.0).meanRelativeErrorTolerance(0.01).stdRelativeErrorTolerance(0.01).meanMinAbsErrorTolerance(0.001).build());
        }


        int count = 1;
        for(TestCase tc : testCases){
            log.info("Starting test case: {} of {}", count, testCases.size());
            log.info("{}", tc);

            Object op = getOp(tc);
            INDArray z = null;
            Nd4j.getRandom().setSeed(tc.getRngSeed());
            if(op instanceof Op) {
                Op o = (Op)op;
                Nd4j.getExecutioner().exec(o);
                z = o.z();
            } else {
                CustomOp o = (CustomOp)op;
                Nd4j.getExecutioner().exec(o);
                z = o.getOutputArgument(0);
            }

            //Check for NaNs, Infs, etc
            int countNaN = Nd4j.getExecutioner().exec(new MatchConditionTransform(z, Nd4j.create(DataType.BOOL, z.shape()), Conditions.isNan())).castTo(DataType.INT).sumNumber().intValue();
            int countInf = Nd4j.getExecutioner().exec(new MatchConditionTransform(z, Nd4j.create(DataType.BOOL, z.shape()), Conditions.isInfinite())).castTo(DataType.INT).sumNumber().intValue();
            assertEquals("NaN - expected 0 values", 0, countNaN);
            assertEquals("Infinite - expected 0 values", 0, countInf);

            //Check min/max values
            double min = z.minNumber().doubleValue();
            if ((tc.isMinValueInclusive() && min < tc.getMinValue()) || (!tc.isMinValueInclusive() && min <= tc.getMinValue())) {
                fail("Minimum value (" + min + ") is less than allowed minimum value (" + tc.getMinValue() + ", inclusive=" + tc.isMinValueInclusive() + "): test case: " + tc);
            }

            double max = z.maxNumber().doubleValue();
            if ((tc.isMaxValueInclusive() && max > tc.getMaxValue()) || (!tc.isMaxValueInclusive() && max >= tc.getMaxValue())) {
                fail("Maximum value (" + max + ") is greater than allowed maximum value (" + tc.getMaxValue() + ", inclusive=" + tc.isMaxValueInclusive() + "): test case: " + tc);
            }

            //Check RNG seed repeatability
            Object op2 = getOp(tc);
            Nd4j.getRandom().setSeed(tc.getRngSeed());
            INDArray z2;
            if(op2 instanceof Op) {
                Op o = (Op)op2;
                Nd4j.getExecutioner().exec(o);
                z2 = o.z();
            } else {
                CustomOp o = (CustomOp)op2;
                Nd4j.getExecutioner().exec(o);
                z2 = o.getOutputArgument(0);
            }
            assertEquals(z, z2);

            //Check mean, stdev
            if(tc.getExpectedMean() != null){
                double mean = z.meanNumber().doubleValue();
                double re = relError(tc.getExpectedMean(), mean);
                double ae = Math.abs(tc.getExpectedMean() - mean);
                if(re > tc.getMeanRelativeErrorTolerance() && (tc.getMeanMinAbsErrorTolerance() == null || ae > tc.getMeanMinAbsErrorTolerance())){
                    fail("Relative error for mean (" + re + ") exceeds maximum (" + tc.getMeanRelativeErrorTolerance() +
                            ") - expected mean = " + tc.getExpectedMean() + " vs. observed mean = " + mean + " - test: " + tc);
                }
            }
            if(tc.getExpectedStd() != null){
                double std = z.std(true).getDouble(0);
                double re = relError(tc.getExpectedStd(), std);
                double ae = Math.abs(tc.getExpectedStd() - std);
                if(re > tc.getStdRelativeErrorTolerance() && (tc.getStdMinAbsErrorTolerance() == null || ae > tc.getStdMinAbsErrorTolerance())){
                    /*
                    //Histogram for debugging
                    INDArray range = Nd4j.create(new double[]{z.minNumber().doubleValue(), z.maxNumber().doubleValue()}).castTo(tc.getDataType());
                    INDArray n = Nd4j.scalar(DataType.INT,100);
                    INDArray out = Nd4j.create(DataType.INT, 100);
                    DynamicCustomOp histogram = DynamicCustomOp.builder("histogram_fixed_width")
                            .addInputs(z, range, n)
                            .addOutputs(out)
                            .build();
                    Nd4j.getExecutioner().exec(histogram);
                    System.out.println(range);
                    System.out.println(out.toString().replaceAll("\\s", ""));
                    */
                    fail("Relative error for stdev (" + re + ") exceeds maximum (" + tc.getStdRelativeErrorTolerance() +
                            ") - expected stdev = " + tc.getExpectedStd() + " vs. observed stdev = " + std + " - test: " + tc);
                }
            }

            count++;
        }


    }

    private static double minValue(DataType dataType){
       switch (dataType){
           case DOUBLE:
               return -Double.MAX_VALUE;
           case FLOAT:
               return -Float.MAX_VALUE;
           case HALF:
               return -65504.0;
           default:
               throw new RuntimeException("Dtype not supported: " + dataType);
       }
    }

    private static double maxValue(DataType dataType){
        switch (dataType){
            case DOUBLE:
                return Double.MAX_VALUE;
            case FLOAT:
                return Float.MAX_VALUE;
            case HALF:
                return 65504.0;
            default:
                throw new RuntimeException("Dtype not supported: " + dataType);
        }
    }


    private static Object getOp(TestCase tc){

        switch (tc.getOpType()){
            //Legacy (non-custom) RNG ops
            case "bernoulli":
                return new BernoulliDistribution(tc.arr(), (double)tc.prop("prob"));
            case "uniform":
                return new UniformDistribution(tc.arr(), tc.prop("min"), tc.prop("max"));
            case "gaussian":
                return new GaussianDistribution(tc.arr(), (double)tc.prop("mean"), tc.prop("std"));
            case "binomial":
                return new BinomialDistribution(tc.arr(), tc.prop("n"), (double)tc.prop("p"));
            case "truncated_normal":
                return new TruncatedNormalDistribution(tc.arr(), (double)tc.prop("mean"), tc.prop("std"));
            case "dropout":
                INDArray z = tc.arr();
                z.assign(1.0);
                return new DropOut(z, tc.prop("p"));
            case "dropout_inverted":
                INDArray z2 = tc.arr();
                z2.assign(1.0);
                return new DropOutInverted(z2, tc.prop("p"));
            case "linspace":
                return new Linspace(tc.arr(), tc.prop("from"), tc.prop("to"));
            case "lognormal":
                return new LogNormalDistribution(tc.arr(), (double)tc.prop("mu"), tc.prop("s"));
            case "choice":
                INDArray source = Nd4j.linspace(0, 10, 11, tc.getDataType());
                INDArray probs = Nd4j.ones(11).divi(11);
                return new Choice(source, probs, tc.arr());
            case "probabilisticmerge":
                INDArray x = Nd4j.zeros(tc.getDataType(), tc.getShape());
                INDArray y = Nd4j.ones(tc.getDataType(), tc.getShape());
                return new ProbablisticMerge(x, y, tc.arr(), tc.prop("prob"));
            case "range":
                double rMin = tc.prop("min");
                double rMax = tc.prop("max");
                double step = (rMax - rMin) / (double) ArrayUtil.prodLong(tc.shape);
                DynamicCustomOp op = DynamicCustomOp.builder("range")
                        .addFloatingPointArguments(rMin, rMax, step)
                        .addOutputs(tc.arr())
                        .build();
                return op;
            case "alphaDropout":
                double alpha = alphaDropoutA(tc.prop("p"));
                double beta = alphaDropoutB(tc.prop("p"));
                return new AlphaDropOut(Nd4j.ones(tc.getDataType(), tc.shape), tc.arr(), tc.prop("p"), alpha, ALPHA_PRIME, beta);


            case "distributionuniform":
                INDArray shape = tc.getShape().length == 0 ? Nd4j.empty(DataType.LONG) : Nd4j.create(ArrayUtil.toDouble(tc.shape)).castTo(DataType.LONG);
                return new DistributionUniform(shape, tc.arr(), tc.prop("min"), tc.prop("max"));
            case "randombernoulli":
                INDArray shape2 = tc.getShape().length == 0 ? Nd4j.empty(DataType.LONG) : Nd4j.create(ArrayUtil.toDouble(tc.shape)).castTo(DataType.LONG);
                return new RandomBernoulli(shape2, tc.arr(), tc.prop("prob"));
            case "randomexponential":
                INDArray shape3 = tc.getShape().length == 0 ? Nd4j.empty(DataType.LONG) : Nd4j.create(ArrayUtil.toDouble(tc.shape)).castTo(DataType.LONG);
                return new RandomExponential(shape3, tc.arr(), tc.prop("lambda"));
            case "randomnormal":
                INDArray shape4 = tc.getShape().length == 0 ? Nd4j.empty(DataType.LONG) : Nd4j.create(ArrayUtil.toDouble(tc.shape)).castTo(DataType.LONG);
                return DynamicCustomOp.builder("randomnormal")
                        .addFloatingPointArguments(tc.prop("mean"), tc.prop("std"))
                        .addInputs(shape4)
                        .addOutputs(tc.arr())
                        .build();
            case "randomstandardnormal":
                INDArray shape5 = tc.getShape().length == 0 ? Nd4j.empty(DataType.LONG) : Nd4j.create(ArrayUtil.toDouble(tc.shape)).castTo(DataType.LONG);
                return new RandomStandardNormal(shape5, Nd4j.create(tc.getDataType(), tc.getShape()));
            default:
                throw new RuntimeException("Not yet implemented: " + tc.getOpType());
        }
    }

    private static double relError(double x, double y){
        return Math.abs(x-y) / (Math.abs(x) + Math.abs(y));
    }


    public static final double DEFAULT_ALPHA =  1.6732632423543772;
    public static final double DEFAULT_LAMBDA = 1.0507009873554804;
    public static final double ALPHA_PRIME = -DEFAULT_LAMBDA * DEFAULT_ALPHA;
    public static double alphaDropoutA(double p){
        return 1.0 / Math.sqrt(p + ALPHA_PRIME*ALPHA_PRIME * p * (1-p));
    }

    public static double alphaDropoutB(double p){
        double alphaPrime = -DEFAULT_LAMBDA * DEFAULT_ALPHA;
        return -alphaDropoutA(p) * (1-p)*alphaPrime;
    }
}
