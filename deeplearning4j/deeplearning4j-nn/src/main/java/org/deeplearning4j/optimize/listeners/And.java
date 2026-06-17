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

import lombok.AllArgsConstructor;
import org.deeplearning4j.nn.api.Model;

import java.util.Arrays;
import java.util.List;

@AllArgsConstructor
public class And extends FailureTrigger{

    protected List<FailureTrigger> triggers;

    public And(FailureTrigger... triggers){
        this.triggers = Arrays.asList(triggers);
    }

    @Override
    public boolean triggerFailure(CallType callType, int iteration, int epoch, Model model) {
        boolean b = true;
        for(FailureTrigger ft : triggers)
            b &= ft.triggerFailure(callType, iteration, epoch, model);
        return b;
    }

    @Override
    public void initialize(){
        super.initialize();
        for(FailureTrigger ft : triggers)
            ft.initialize();
    }
}
