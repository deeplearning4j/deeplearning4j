/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.schedule;

/**
 * A "Wrapper" schedule that ramps up from {@code 1/numIter * baseLR} to {@code baseLR} over numIter iterations.
 * The base learning rate is determined by the underlying ISchedule, as a function of time.
 * This can be used to provide a slow start, for use cases such as transfer learning.
 *
 * @author Alex Black
 */
public class RampSchedule implements ISchedule {

    protected final ISchedule baseSchedule;
    protected final int numIter;

    public RampSchedule(ISchedule baseSchedule, int numIter){
        this.baseSchedule = baseSchedule;
        this.numIter = numIter;
    }

    @Override
    public double valueAt(int iteration, int epoch) {
        double base = baseSchedule.valueAt(iteration, epoch);
        if(iteration >= numIter - 1){
            return base;
        }
        double frac = (iteration+1) / (double)numIter;
        return frac * base;
    }

    @Override
    public ISchedule clone() {
        return new RampSchedule(baseSchedule.clone(), numIter);
    }
}
