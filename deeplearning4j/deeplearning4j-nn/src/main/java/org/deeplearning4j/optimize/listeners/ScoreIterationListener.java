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

package org.deeplearning4j.optimize.listeners;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

/**
 * Score iteration listener
 *
 * @author Adam Gibson
 */
@Slf4j
public class ScoreIterationListener extends BaseTrainingListener implements Serializable {
    private int printIterations = 10;

    /**
     * @param printIterations    frequency with which to print scores (i.e., every printIterations parameter updates)
     */
    public ScoreIterationListener(int printIterations) {
        this.printIterations = printIterations;
    }

    /** Default constructor printing every 10 iterations */
    public ScoreIterationListener() {}

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if (printIterations <= 0)
            printIterations = 1;
        if (iteration % printIterations == 0) {
            double score = model.score();
            log.info("Score at iteration {} is {}", iteration, score);
        }
    }

    @Override
    public String toString(){
        return "ScoreIterationListener(" + printIterations + ")";
    }
}
