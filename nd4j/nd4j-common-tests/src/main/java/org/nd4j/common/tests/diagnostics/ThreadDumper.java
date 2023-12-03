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
package org.nd4j.common.tests.diagnostics;

import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class ThreadDumper {

    public static ScheduledExecutorService ses = Executors.newScheduledThreadPool(1);


    public static void printThreadDumpsPeriodically(long everyMs) {
        ses.scheduleAtFixedRate(() -> printDump(), 1, everyMs, TimeUnit.MILLISECONDS);

    }

    public static void printDump() {
        for (Map.Entry<Thread, StackTraceElement[]> entry : Thread.getAllStackTraces().entrySet()) {
            System.out.println(entry.getKey() + " " + entry.getKey().getState());
            for (StackTraceElement ste : entry.getValue()) {
                System.out.println("\tat " + ste);
            }
            System.out.println();
        }
    }



}
