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

package org.deeplearning4j.optimize.listeners;

import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;


import java.io.Serializable;

/**
 * A simple listener that collects scores to a list every N iterations. Can also optionally log the score.
 *
 * @author Alex Black
 */
@Data
@Slf4j
public class CollectScoresListener extends BaseTrainingListener implements Serializable {

    private final int frequency;
    private final boolean logScore;
    private final IntArrayList listIteration;
    private final DoubleArrayList listScore;

    public CollectScoresListener(int frequency) {
        this(frequency, false);
    }

    public CollectScoresListener(int frequency, boolean logScore){
        this.frequency = frequency;
        this.logScore = logScore;
        listIteration = new IntArrayList();
        listScore = new DoubleArrayList();
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if(iteration % frequency == 0){
            double score = model.score();
            listIteration.add(iteration);
            listScore.add(score);
            if(logScore) {
                log.info("Score at iteration {} is {}", iteration, score);
            }
        }
    }
}
