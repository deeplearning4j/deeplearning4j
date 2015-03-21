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

package org.nd4j.linalg.api.ops;


import org.nd4j.linalg.api.complex.IComplexNumber;

import java.util.List;

/**
 * An accumulation is an op that given:
 * x -> the origin ndarray
 * y -> the pairwise ndarray
 * n -> the number of times to accumulate
 * <p/>
 * <p/>
 * Of note here in the extra arguments.
 * <p/>
 * An accumulation (or reduction in some terminology)
 * has a concept of a starting value.
 * <p/>
 * The starting value is the initialization of the solution
 * to the operation.
 * <p/>
 * An accumulation should always have the extraArgs()
 * contain the zero value as the first value.
 * <p/>
 * This allows the architecture to generalize to different backends
 * and gives the implementer of a backend a way of hooking in to
 * passing parameters to different engines.
 *
 * @author Adam Gibson
 */
public interface Accumulation extends Op {
    /**
     * Current accumulated result
     *
     * @return
     */
    IComplexNumber currentResultComplex();

    /**
     * Current result
     *
     * @return
     */
    Number currentResult();

    /**
     * Update the current result to be this result
     *
     * @param result the result
     */
    void update(Number result);

    /**
     * Update the current result to be this result
     *
     * @param result the result
     */
    void update(IComplexNumber result);


    /**
     * Initial value
     *
     * @return the initial value
     */
    Number zero();

    /**
     * Complex initial value
     *
     * @return the complex initial value
     */
    IComplexNumber zeroComplex();


    /**
     * Other accmuluations from the primary
     *
     * @return other accumulations from the primary
     */
    List<IComplexNumber> otherAccumComplex();

    /**
     * Other accmuluations from the primary
     *
     * @return other accumulations from the primary
     */
    List<Number> otherAccum();

    /**
     * Set the current result
     *
     * @param number the result
     */
    void setCurrentResult(Number number);

    /**
     * Set the current complex number
     * result
     *
     * @param complexNumber the current complex number
     *                      result
     */
    void setCurrentResultComplex(IComplexNumber complexNumber);

}
