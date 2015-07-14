/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 * Originally based on cc.mallet.optimize.ConjugateGradient
 * 
 * Rewritten based on Conjugate Gradient algorithm in Bengio et al.,
 * Deep Learning (in preparation) Ch8
 *
 */

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
    	//line is current gradient
    	//Last gradient is stored in searchState map
    	INDArray gLast = (INDArray) searchState.get(GRADIENT_KEY);		//Previous iteration gradient
        INDArray searchDirLast = (INDArray) searchState.get(SEARCH_DIR);//Previous iteration search dir
        
        //Calculate gamma (or beta, by Bengio et al. notation). Polak and Ribiere method.
        // = ((grad(current)-grad(last)) \dot (grad(current))) / (grad(last) \dot grad(last))
        double dgg = Nd4j.getBlasWrapper().dot(line.sub(gLast),line);
        double gg = Nd4j.getBlasWrapper().dot(gLast, gLast);
        double gamma = dgg / gg;

        //Compute search direction:
        //searchDir = -gradient + gamma * searchDirLast
        INDArray searchDir = line.neg().addi(searchDirLast.muli(gamma));

        //Store current gradient and search direction for
        //(a) use in BaseOptimizer.optimize(), and (b) next iteration
        searchState.put(GRADIENT_KEY, line);
        searchState.put(SEARCH_DIR, searchDir);
    }

    @Override
    public void postStep() {
    	//no-op
    }

    @Override
    public void setupSearchState(Pair<Gradient, Double> pair) {
        super.setupSearchState(pair);
        INDArray gradient = (INDArray) searchState.get(GRADIENT_KEY);
        INDArray params = model.params();
        
        searchState.put(SEARCH_DIR,Nd4j.zeros(params.shape()));	//Initialize to 0, as per Bengio et al.
        searchState.put(PARAMS_KEY, params);
        searchState.put(GRADIENT_KEY, gradient);	//Consequence: on first iteration, gamma = 0
    }



}
