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

package org.deeplearning4j.optimize.api;


import org.deeplearning4j.nn.api.Model;

import java.io.Serializable;

@Deprecated
public abstract class IterationListener extends BaseTrainingListener implements Serializable {

    /**
     * Event listener for each iteration
     * @param iteration the iteration
     * @param model the model iterating
     */
    public abstract void iterationDone(Model model, int iteration, int epoch);

}
