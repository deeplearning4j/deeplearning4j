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

package org.deeplearning4j.optimize.listeners;

import lombok.Data;
import org.deeplearning4j.nn.api.Model;

import java.io.Serializable;

@Data
public abstract class FailureTrigger implements Serializable {

    private static final long serialVersionUID = 1L;

    private boolean initialized = false;

    /**
     * If true: trigger the failure. If false: don't trigger failure
     * @param callType  Type of call
     * @param iteration Iteration number
     * @param epoch     Epoch number
     * @param model     Model
     * @return
     */
    public abstract boolean triggerFailure(CallType callType, int iteration, int epoch, Model model);

    public boolean initialized(){
        return initialized;
    }

    public void initialize(){
        this.initialized = true;
    }
}
