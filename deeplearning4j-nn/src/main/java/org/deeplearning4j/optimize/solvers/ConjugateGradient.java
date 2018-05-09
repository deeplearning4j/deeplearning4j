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
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.api.TerminationCondition;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;


/**Originally based on cc.mallet.optimize.ConjugateGradient
 * 
 * Rewritten based on Conjugate Gradient algorithm in Bengio et al.,
 * Deep Learning (in preparation) Ch8.
 * See also Nocedal & Wright, Numerical optimization, Ch5
 */
public class ConjugateGradient extends BaseOptimizer {
    private static final long serialVersionUID = -1269296013474864091L;
    private static final Logger logger = LoggerFactory.getLogger(ConjugateGradient.class);

    public ConjugateGradient(NeuralNetConfiguration conf, StepFunction stepFunction,
                    Collection<TrainingListener> trainingListeners, Model model) {
        super(conf, stepFunction, trainingListeners, model);
    }


    public ConjugateGradient(NeuralNetConfiguration conf, StepFunction stepFunction,
                    Collection<TrainingListener> trainingListeners,
                    Collection<TerminationCondition> terminationConditions, Model model) {
        super(conf, stepFunction, trainingListeners, terminationConditions, model);
    }

    @Override
    public void preProcessLine() {
        INDArray gradient = (INDArray) searchState.get(GRADIENT_KEY);
        INDArray searchDir = (INDArray) searchState.get(SEARCH_DIR);
        if (searchDir == null)
            searchState.put(SEARCH_DIR, gradient);
        else
            searchDir.assign(gradient);
    }

    @Override
    public void postStep(INDArray gradient) {
        //line is current gradient
        //Last gradient is stored in searchState map
        INDArray gLast = (INDArray) searchState.get(GRADIENT_KEY); //Previous iteration gradient
        INDArray searchDirLast = (INDArray) searchState.get(SEARCH_DIR);//Previous iteration search dir

        //Calculate gamma (or beta, by Bengio et al. notation). Polak and Ribiere method.
        // = ((grad(current)-grad(last)) \dot (grad(current))) / (grad(last) \dot grad(last))
        double dgg = Nd4j.getBlasWrapper().dot(gradient.sub(gLast), gradient);
        double gg = Nd4j.getBlasWrapper().dot(gLast, gLast);
        double gamma = Math.max(dgg / gg, 0.0);
        if (dgg <= 0.0)
            logger.debug("Polak-Ribiere gamma <= 0.0; using gamma=0.0 -> SGD line search. dgg={}, gg={}", dgg, gg);

        //Standard Polak-Ribiere does not guarantee that the search direction is a descent direction
        //But using max(gamma_Polak-Ribiere,0) does guarantee a descent direction. Hence the max above.
        //See Nocedal & Wright, Numerical Optimization, Ch5
        //If gamma==0.0, this is equivalent to SGD line search (i.e., search direction == negative gradient)

        //Compute search direction:
        //searchDir = gradient + gamma * searchDirLast
        INDArray searchDir = searchDirLast.muli(gamma).addi(gradient);

        //Store current gradient and search direction for
        //(a) use in BaseOptimizer.optimize(), and (b) next iteration
        searchState.put(GRADIENT_KEY, gradient);
        searchState.put(SEARCH_DIR, searchDir);
    }



}
