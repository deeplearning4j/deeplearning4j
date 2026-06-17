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
import lombok.NonNull;
import org.deeplearning4j.nn.api.Model;

@Data
public class UserNameTrigger extends FailureTrigger {
    private final String userName;
    private boolean shouldFail = false;

    public UserNameTrigger(@NonNull String userName) {
        this.userName = userName;
    }


    @Override
    public boolean triggerFailure(CallType callType, int iteration, int epoch, Model model) {
        return shouldFail;
    }

    @Override
    public void initialize(){
        super.initialize();
        shouldFail = this.userName.equalsIgnoreCase(System.getProperty("user.name"));
    }
}
