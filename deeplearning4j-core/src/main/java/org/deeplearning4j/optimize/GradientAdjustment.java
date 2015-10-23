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

package org.deeplearning4j.optimize;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.AdaGrad;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Gradient adjustment
 * @author Adam Gibson
 */

public class GradientAdjustment {

    private GradientAdjustment(){}


    private static final Logger log = LoggerFactory.getLogger(GradientAdjustment.class);


    /**
     * Updates each variable wrt its gradient
     * @param iteration the iteration
     * @param batchSize the batch size of the input
     * @param conf the configuration
     * @param params the model params
     * @param gradient the gradients for the variables
     * @param adaGrad the adagrad map (per variable adagrad entries(
     * @return updated gradient
     */
    @Deprecated
    public static void updateGradientAccordingToParams(int iteration, int batchSize, NeuralNetConfiguration conf, INDArray params,
                                                           INDArray gradient, GradientUpdater adaGrad, INDArray lastStep, String paramType) {
        if(adaGrad == null)
            adaGrad = new AdaGrad(conf.getLayer().getLearningRate());


        if(lastStep == null)
            lastStep = Nd4j.ones((params.shape()));

        //change up momentum after so many iterations if specified
        double momentum = conf.getLayer().getMomentum();
        if(conf.getLayer().getMomentumAfter() != null && !conf.getLayer().getMomentumAfter().isEmpty()) {
            int key = conf.getLayer().getMomentumAfter().keySet().iterator().next();
            if(iteration >= key) {
                momentum = conf.getLayer().getMomentumAfter().get(key);
            }
        }

        //RMSPROP
        if(conf.getLayer().getRmsDecay() > 0) {
            lastStep.assign(lastStep.mul(conf.getLayer().getRmsDecay()).addi(Transforms.pow(gradient,2).muli((1 - conf.getLayer().getRmsDecay()))));
            gradient = gradient.mul(conf.getLayer().getLearningRate()).negi().divi(Transforms.sqrt(lastStep.add(Nd4j.EPS_THRESHOLD)));
        }

        //calculate gradient
        gradient = adaGrad.getGradient(gradient,0);


        //apply nesterov's AFTER learning rate update
        if (momentum > 0) {
            gradient = lastStep.mul(momentum).subi(gradient);
            //in place update on the step cache
            lastStep.assign(gradient);
        }

        //simulate post gradient application  and apply the difference to the gradient to decrease the change the gradient has
        if(conf.isUseRegularization() && conf.getLayer().getL2() > 0 && !(gradient.equals(DefaultParamInitializer.BIAS_KEY)))
            gradient.subi(params.mul(conf.getLayer().getL2()));
        else if(conf.isUseRegularization() && conf.getLayer().getL1() < 0 && !(gradient.equals(DefaultParamInitializer.BIAS_KEY)))
            gradient.subi(Transforms.sign(params).muli(conf.getLayer().getL1()));


        if(conf.isConstrainGradientToUnitNorm())
            gradient.divi(gradient.norm2(Integer.MAX_VALUE));

        gradient.divi(batchSize);


    }


}
