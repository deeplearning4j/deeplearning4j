/*-
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

package org.deeplearning4j.optimize.solvers;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.api.TerminationCondition;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;

/**
 * LBFGS
 * @author Adam Gibson
 */
public class LBFGS extends BaseOptimizer {
    private static final long serialVersionUID = 9148732140255034888L;
    private int m = 4;

    public LBFGS(NeuralNetConfiguration conf, StepFunction stepFunction,
                    Collection<TrainingListener> trainingListeners, Model model) {
        super(conf, stepFunction, trainingListeners, model);
    }

    public LBFGS(NeuralNetConfiguration conf, StepFunction stepFunction,
                    Collection<TrainingListener> trainingListeners,
                    Collection<TerminationCondition> terminationConditions, Model model) {
        super(conf, stepFunction, trainingListeners, terminationConditions, model);
    }

    @Override
    public void setupSearchState(Pair<Gradient, Double> pair) {
        super.setupSearchState(pair);
        INDArray params = (INDArray) searchState.get(PARAMS_KEY);
        searchState.put("s", new LinkedList<INDArray>()); // holds parameters differences
        searchState.put("y", new LinkedList<INDArray>()); // holds gradients differences
        searchState.put("rho", new LinkedList<Double>());
        searchState.put("oldparams", params.dup());

    }

    @Override
    public void preProcessLine() {
        if (!searchState.containsKey(SEARCH_DIR))
            searchState.put(SEARCH_DIR, ((INDArray) searchState.get(GRADIENT_KEY)).dup());
    }

    // Numerical Optimization (Nocedal & Wright) section 7.2
    // s = parameters differences (old & current)
    // y = gradient differences (old & current)
    // gamma = initial Hessian approximation (i.e., equiv. to gamma*IdentityMatrix for Hessian)
    // rho = scalar. rho_i = 1/(y_i \dot s_i)
    @Override
    public void postStep(INDArray gradient) {
        INDArray previousParameters = (INDArray) searchState.get("oldparams");
        INDArray parameters = model.params();
        INDArray previousGradient = (INDArray) searchState.get(GRADIENT_KEY);

        LinkedList<Double> rho = (LinkedList<Double>) searchState.get("rho");
        LinkedList<INDArray> s = (LinkedList<INDArray>) searchState.get("s");
        LinkedList<INDArray> y = (LinkedList<INDArray>) searchState.get("y");

        double sy = Nd4j.getBlasWrapper().dot(previousParameters, previousGradient) + Nd4j.EPS_THRESHOLD;
        double yy = Nd4j.getBlasWrapper().dot(previousGradient, previousGradient) + Nd4j.EPS_THRESHOLD;

        INDArray sCurrent;
        INDArray yCurrent;
        if (s.size() >= m) {
            //Optimization: Remove old (no longer needed) INDArrays, and use assign for re-use.
            //Better to do this: fewer objects created -> less memory overall + less garbage collection
            sCurrent = s.removeLast();
            yCurrent = y.removeLast();
            rho.removeLast();
            sCurrent.assign(parameters).subi(previousParameters);
            yCurrent.assign(gradient).subi(previousGradient);
        } else {
            //First few iterations. Need to allocate new INDArrays for storage (via copy operation sub)
            sCurrent = parameters.sub(previousParameters);
            yCurrent = gradient.sub(previousGradient);
        }

        rho.addFirst(1.0 / sy); //Most recent first
        s.addFirst(sCurrent); //Most recent first. si = currParams - oldParams
        y.addFirst(yCurrent); //Most recent first. yi = currGradient - oldGradient

        //assert (s.size()==y.size()) : "Gradient and parameter sizes are not equal";
        if (s.size() != y.size())
            throw new IllegalStateException("Gradient and parameter sizes are not equal");

        //In general: have m elements in s,y,rho.
        //But for first few iterations, have less.
        int numVectors = Math.min(m, s.size());

        double[] alpha = new double[numVectors];

        // First work backwards, from the most recent difference vectors
        Iterator<INDArray> sIter = s.iterator();
        Iterator<INDArray> yIter = y.iterator();
        Iterator<Double> rhoIter = rho.iterator();

        //searchDir: first used as equivalent to q as per N&W, then later used as r as per N&W.
        //Re-using existing array for performance reasons
        INDArray searchDir = (INDArray) searchState.get(SEARCH_DIR);
        searchDir.assign(gradient);

        for (int i = 0; i < numVectors; i++) {
            INDArray si = sIter.next();
            INDArray yi = yIter.next();
            double rhoi = rhoIter.next();

            if (si.length() != searchDir.length())
                throw new IllegalStateException("Gradients and parameters length not equal");

            alpha[i] = rhoi * Nd4j.getBlasWrapper().dot(si, searchDir);
            Nd4j.getBlasWrapper().level1().axpy(searchDir.length(), -alpha[i], yi, searchDir); //q = q-alpha[i]*yi
        }

        //Use Hessian approximation initialization scheme
        //searchDir = H0*q = (gamma*IdentityMatrix)*q = gamma*q
        double gamma = sy / yy;
        searchDir.muli(gamma);

        //Reverse iterators: end to start. Java LinkedLists are doubly-linked,
        // so still O(1) for reverse iteration operations.
        sIter = s.descendingIterator();
        yIter = y.descendingIterator();
        rhoIter = rho.descendingIterator();
        for (int i = 0; i < numVectors; i++) {
            INDArray si = sIter.next();
            INDArray yi = yIter.next();
            double rhoi = rhoIter.next();

            double beta = rhoi * Nd4j.getBlasWrapper().dot(yi, searchDir); //beta = rho_i * y_i^T * r
            //r = r + s_i * (alpha_i - beta)
            Nd4j.getBlasWrapper().level1().axpy(gradient.length(), alpha[i] - beta, si, searchDir);
        }

        previousParameters.assign(parameters);
        previousGradient.assign(gradient); //Update gradient. Still in searchState map keyed by GRADIENT_KEY
    }
}
