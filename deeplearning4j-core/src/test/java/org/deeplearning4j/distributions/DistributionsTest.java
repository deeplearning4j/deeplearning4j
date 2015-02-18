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

package org.deeplearning4j.distributions;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.junit.Test;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

/**
 * @author Adam Gibson
 */
public class DistributionsTest {

    private RandomGenerator rng = new MersenneTwister(123);
    private static final Logger log = LoggerFactory.getLogger(DistributionsTest.class);
    @Test
    public void uniformTest() {

        RealDistribution dist = Distributions.uniform(rng,4);
        int numSamples = 10;
        double avg = 0.0;
        for(int i = 0;i < numSamples; i++)
            avg += dist.sample();

       avg /= numSamples;

        log.info("Avg " + avg);


    }

}
