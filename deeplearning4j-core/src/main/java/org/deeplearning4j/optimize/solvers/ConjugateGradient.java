/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

/**
 @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package org.deeplearning4j.optimize.solvers;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.indexing.conditions.Or;
import org.nd4j.linalg.indexing.functions.Value;
import static org.nd4j.linalg.ops.transforms.Transforms.*;
import org.nd4j.linalg.util.LinAlgExceptions;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;


/**
 * Modified based on cc.mallet.optimize.ConjugateGradient <p/>
 * no termination when zero tolerance 
 *
 * @author Adam Gibson
 * @since 2013-08-25
 */

// Conjugate Gradient, Polak and Ribiere version
// from "Numeric Recipes in C", Section 10.6.

public class ConjugateGradient extends BaseOptimizer {
    private static final Logger logger = LoggerFactory.getLogger(ConjugateGradient.class);



    public ConjugateGradient(NeuralNetConfiguration conf, StepFunction stepFunction, Collection<IterationListener> iterationListeners, Model model) {
        super(conf, stepFunction, iterationListeners, model);
    }


    public ConjugateGradient(NeuralNetConfiguration conf, StepFunction stepFunction, Collection<IterationListener> iterationListeners, Collection<TerminationCondition> terminationConditions, Model model) {
        super(conf, stepFunction, iterationListeners, terminationConditions, model);
    }




    @Override
    public void preProcessLine(INDArray line) {
        //no-op
    }

    @Override
    public void postStep() {
        INDArray g = (INDArray) searchState.get(GRADIENT_KEY);
        INDArray xi = (INDArray) searchState.get("xi");
        INDArray h = (INDArray) searchState.get("h");
        searchState.put("gg",pow(g, 2).sum(Integer.MAX_VALUE).getDouble(0));
        searchState.put("dgg",xi.mul(xi.sub(g)).sum(Integer.MAX_VALUE).getDouble(0));


        double dgg = (double) searchState.get("dgg");
        double gg = (double) searchState.get("gg");
        double gam = dgg / gg;
        searchState.put("gam",gam);
        if(h == null)
            h = g;

        g.assign(xi);
        h.assign(h.mul(gam).addi(xi));


        BooleanIndexing.applyWhere(h, new Or(Conditions.isNan(),Conditions.isInfinite()), new Value(Nd4j.EPS_THRESHOLD));

        // gdruck
        // Mallet line search algorithms stop search whenever
        // a step is found that increases the value significantly.
        // ConjugateGradient assumes that line maximization finds something
        // close
        // to the maximum in that direction. In tests, sometimes the
        // direction suggested by CG was downhill. Consequently, here I am
        // setting the search direction to the gradient if the slope is
        // negative or 0.
        if (Nd4j.getBlasWrapper().dot(xi, h) > 0)
            xi.assign(h);
        else {
            logger.warn("Reverting back to GA");
            h.assign(xi);
        }

        searchState.put(GRADIENT_KEY,g);
        searchState.put("xi",xi);
        searchState.put("h",xi.add(h.mul(gam)));

    }

    @Override
    public void setupSearchState(Pair<Gradient, Double> pair) {
        super.setupSearchState(pair);
        INDArray gradient = (INDArray) searchState.get(GRADIENT_KEY);
        searchState.put("h",gradient.dup());
        searchState.put("xi",gradient.dup());
        searchState.put("gg",0.0);
        searchState.put("gam",0.0);
        searchState.put("dgg",0.0);

    }


}
