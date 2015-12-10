package org.arbiter.optimize.api.saving;

import org.arbiter.optimize.api.OptimizationResult;

import java.io.IOException;

/**Idea: We can't store all results in memory in general (might have thousands of candidates with millions of
 * parameters each)
 * So instead: return a reference to the saved result. Idea is that the result may be saved to disk or a database,
 * and we can easily load it back into memory (if required) using the methods here
 */
public interface ResultReference<T,M> {

    OptimizationResult<T,M> getResult() throws IOException;

}
