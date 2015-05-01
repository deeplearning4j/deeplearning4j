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

package org.deeplearning4j.scaleout.aggregator;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.scaleout.job.Job;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * INDArray averager
 *
 * @author Adam Gibson
 */
public class INDArrayAggregator extends WorkAccumulator {
    private INDArray averaged;
    private static final Logger log = LoggerFactory.getLogger(INDArrayAggregator.class);

    @Override
    public void accumulate(Job toAccumulate) {
        if(toAccumulate.getResult() == null || !(toAccumulate.getResult() instanceof INDArray)) {
            log.warn("Not accumulating result: must be of type INDArray and not null");
            return;
        }

        INDArray arr = (INDArray) toAccumulate.getResult();
        seenSoFar++;
        if(averaged == null) {
            this.averaged = arr;
        }

        else {
            averaged.addi(arr);
        }
    }

    @Override
    public Job aggregate() {
        if(averaged == null)
            return empty();
        Job ret =  new Job(averaged.div(seenSoFar),"");
        seenSoFar = 0.0;
        return ret;
    }

    @Override
    public void init(Configuration conf) {
        //no-op
    }
}
