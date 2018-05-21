/*-
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

package org.deeplearning4j.optimize.api;

import java.io.Serializable;

/**
 * Created by agibsonccc on 12/24/14.
 */
public interface TerminationCondition extends Serializable {

    /**
     * Whether to terminate based on the given metadata
     * @param cost the new cost
     * @param oldCost the old cost
     * @param otherParams
     * @return
     */
    boolean terminate(double cost, double oldCost, Object[] otherParams);

}
