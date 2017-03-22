/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.deeplearning4j.arbiter.optimize.runner.listener.runner;

import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;

/**Status listener: is registered with the IOptimizationRunner, and receives callbacks whenever events occur
 */
public interface OptimizationRunnerStatusListener {

    /** Called when optimization runner starts execution */
    void onInitialization(IOptimizationRunner runner);

    /** Called when optimization runner terminates */
    void onShutdown(IOptimizationRunner runner);

    /** Called when any of the summary stats change: number scheduled, number completed, number failed,
     * best score, etc. */
    void onStatusChange(IOptimizationRunner runner);

    /** On completion of an optimization task - due to successful execution, failure, or being cancelled etc.*/
    void onCompletion(OptimizationResult<?, ?, ?> result);

}
