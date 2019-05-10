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

package org.deeplearning4j.spark.time;

/**
 * A time source is an abstraction of system time away from the local system clock.
 * Typically it is used in distributed computing settings, to allow for different time implementations, such as NTP
 * over the internet (via {@link NTPTimeSource}, local synchronization (LAN only - not implemented), or simply using the
 * standard clock on each machine (System.currentTimeMillis() via {@link SystemClockTimeSource}.
 *
 * @author Alex Black
 */
public interface TimeSource {

    /**
     * Get the current time in milliseconds, according to this TimeSource
     * @return Current time, since epoch
     */
    long currentTimeMillis();

    //TODO add methods related to accuracy etc

}
