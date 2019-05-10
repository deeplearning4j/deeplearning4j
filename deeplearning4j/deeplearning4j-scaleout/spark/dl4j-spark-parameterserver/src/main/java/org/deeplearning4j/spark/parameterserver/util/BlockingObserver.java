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

package org.deeplearning4j.spark.parameterserver.util;

import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.util.Observable;
import java.util.Observer;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author raver119@gmail.com
 */
@Slf4j
@Data
public class BlockingObserver implements Observer {
    protected AtomicBoolean state = new AtomicBoolean(false);
    protected AtomicBoolean exception;

    public BlockingObserver(AtomicBoolean exception){
        this.exception = exception;
    }

    @Override
    public void update(Observable o, Object arg) {
        state.set(true);
        //notify();
    }

    /**
     * This method blocks until state is set to True
     */
    public void waitTillDone() throws InterruptedException {
        while (!exception.get() && !state.get()) {
            //LockSupport.parkNanos(1000L);
            // we don't really need uber precision here, sleep is ok
            Thread.sleep(5);
        }
    }
}
