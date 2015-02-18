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

package org.nd4j.linalg.solvers.api;


/**
 * Optimizer that handles optimizing parameters. Handles line search
 * and all the components involved with early stopping
 */
public interface OptimizerMatrix {

    /**
     * Run optimize
     *
     * @return whether the algorithm converged properly
     */
    public boolean optimize();

    /**
     * Run optimize up to the specified number of epochs
     *
     * @param numIterations the max number of epochs to run
     * @return whether the algorihtm converged properly
     */
    public boolean optimize(int numIterations);

    /**
     * Whether the algorithm is converged
     *
     * @return true if the algorithm converged, false otherwise
     */
    public boolean isConverged();

    /**
     * The default max number of iterations to run
     *
     * @param maxIterations
     */
    void setMaxIterations(int maxIterations);

    /**
     * The tolerance for change when running
     *
     * @param tolerance
     */
    void setTolerance(double tolerance);


}
