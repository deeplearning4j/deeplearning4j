/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.parameterserver.distributed.logic.retransmission;

import lombok.NonNull;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.logic.RetransmissionHandler;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.transport.Transport;

/**
 * @author raver119@gmail.com
 */
@Deprecated
public class DefaultRetransmissionHandler implements RetransmissionHandler {
    private VoidConfiguration configuration;

    @Override
    public void init(@NonNull VoidConfiguration configuration, Transport transport) {
        this.configuration = configuration;
    }

    @Override
    public void onBackPressure() {
        try {
            Thread.sleep(2000);
        } catch (Exception e) {
        }
    }

    @Override
    public void handleMessage(TrainingMessage message) {

    }
}
