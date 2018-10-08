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

package org.deeplearning4j.util;

import java.util.concurrent.locks.LockSupport;

/**
 * Utils for the basic use and flow of threads.
 */
public class ThreadUtils {
    public static void uncheckedSleep(long millis) {
        LockSupport.parkNanos(millis * 1000000);
        // we must check the interrupted status in case this is used in a loop
        // Otherwise we may end up spinning 100% without breaking out on an interruption
        if (Thread.currentThread().isInterrupted()) {
            throw new UncheckedInterruptedException();
        }
    }

    public static void uncheckedSleepNanos(long nanos) {
        LockSupport.parkNanos(nanos);
        // we must check the interrupted status in case this is used in a loop
        // Otherwise we may end up spinning 100% without breaking out on an interruption
        if (Thread.currentThread().isInterrupted()) {
            throw new UncheckedInterruptedException();
        }
    }
    
    /**
     * Similar to {@link InterruptedException} in concept, but unchecked.  Allowing this to be thrown without being 
     * explicitly declared in the API.
     */
    public static class UncheckedInterruptedException extends RuntimeException {
	
    }
}
