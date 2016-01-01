package org.arbiter.optimize.runner.listener.runner;

import org.arbiter.optimize.api.OptimizationResult;
import org.arbiter.optimize.runner.IOptimizationRunner;

/**Status listener: is registered with the OptimizationRunner, and recieves callbacks whenever events occur
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
