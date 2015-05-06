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



import org.deeplearning4j.scaleout.job.Job;


/**
 * Parameter averaging algorithm.
 * It handles summing and averaging over all of the results
 * accumulated so far
 */
public abstract class WorkAccumulator implements JobAggregator {

    private Job averaged = null;
    protected double seenSoFar = 0.0;




    protected Job empty() {
        return new Job(null,"");
    }

    /**
     * Averages the results of the network so far
     * @param toAccumulate the network to average in
     */
    public abstract void accumulate(Job toAccumulate);



    /**
     * The averaged network
     * @return the averaged network
     */
    public Job averaged() {
        return averaged;
    }

}
