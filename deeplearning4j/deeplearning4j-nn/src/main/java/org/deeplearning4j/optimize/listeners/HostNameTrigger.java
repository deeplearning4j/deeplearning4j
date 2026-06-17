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
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;

import java.net.InetAddress;

@Data
@Slf4j
public class HostNameTrigger extends FailureTrigger{
    private final String hostName;
    private boolean shouldFail = false;

    public HostNameTrigger(@NonNull String hostName) {
        this.hostName = hostName;
    }


    @Override
    public boolean triggerFailure(CallType callType, int iteration, int epoch, Model model) {
        return shouldFail;
    }

    @Override
    public void initialize(){
        super.initialize();
        try {
            String hostname = InetAddress.getLocalHost().getHostName();
            log.info("FailureTestingListere hostname: {}", hostname);
            shouldFail = this.hostName.equalsIgnoreCase(hostname);
        } catch (Exception e){
            throw new RuntimeException(e);
        }
    }
}
