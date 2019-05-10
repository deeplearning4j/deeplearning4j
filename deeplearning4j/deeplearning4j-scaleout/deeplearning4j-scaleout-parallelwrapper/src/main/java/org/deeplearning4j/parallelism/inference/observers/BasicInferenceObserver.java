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

package org.deeplearning4j.parallelism.inference.observers;

import lombok.extern.slf4j.Slf4j;

import java.util.Observable;
import java.util.Observer;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;

/**
 * Simple Observer implementation for
 * sequential inference
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class BasicInferenceObserver implements Observer {
    private AtomicBoolean finished;

    public BasicInferenceObserver() {
        finished = new AtomicBoolean(false);
    }

    @Override
    public void update(Observable o, Object arg) {
        finished.set(true);
    }

    /**
     * FOR DEBUGGING ONLY, TO BE REMOVED BEFORE MERGE
     */
    public void waitTillDone() {
        while (!finished.get()) {
            LockSupport.parkNanos(1000);
        }
    }
}
