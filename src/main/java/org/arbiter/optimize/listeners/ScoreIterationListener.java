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

package org.arbiter.optimize.listeners;

import org.arbiter.nn.api.Model;
import org.arbiter.optimize.api.IterationListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Score iteration listener
 *
 * @author Adam Gibson
 */
public class ScoreIterationListener implements IterationListener {
    private int printIterations = 10;
    private static final Logger log = LoggerFactory.getLogger(ScoreIterationListener.class);

    public ScoreIterationListener(int printIterations) {
        this.printIterations = printIterations;
    }

    public ScoreIterationListener() {
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        if(iteration % printIterations == 0)
            log.info("Score at iteration " + iteration + " is " + model.score());

    }
}
