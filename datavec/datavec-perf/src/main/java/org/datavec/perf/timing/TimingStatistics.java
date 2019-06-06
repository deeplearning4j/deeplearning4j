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

package org.datavec.perf.timing;

import lombok.Builder;
import lombok.Data;


/**
 * The timing statistics for a data pipeline including:
 * ndarray creation time in nanoseconds
 * disk reading time in nanoseconds
 * bandwidth used in host to device in nano seconds
 * bandwidth device to host in nanoseconds
 *
 * @author Adam Gibson
 */
@Builder
@Data
public class TimingStatistics {

    private long ndarrayCreationTimeNanos;
    private long diskReadingTimeNanos;
    private long bandwidthNanosHostToDevice;
    private long bandwidthDeviceToHost;


    /**
     * Accumulate the given statistics
     * @param timingStatistics the statistics to add
     * @return the added statistics
     */
    public TimingStatistics add(TimingStatistics timingStatistics) {
        return TimingStatistics.builder()
                .ndarrayCreationTimeNanos(ndarrayCreationTimeNanos + timingStatistics.ndarrayCreationTimeNanos)
                .bandwidthNanosHostToDevice(bandwidthNanosHostToDevice + timingStatistics.bandwidthNanosHostToDevice)
                .diskReadingTimeNanos(diskReadingTimeNanos + timingStatistics.diskReadingTimeNanos)
                .bandwidthDeviceToHost(bandwidthDeviceToHost + timingStatistics.bandwidthDeviceToHost)
                .build();
    }


    /**
     * Average the results relative to the number of n.
     * This method is meant to be used alongside
     * {@link #add(TimingStatistics)}
     * accumulated a number of times
     * @param n n the number of elements
     * @return the averaged results
     */
    public TimingStatistics average(long n) {
        return TimingStatistics.builder()
                .ndarrayCreationTimeNanos(ndarrayCreationTimeNanos / n)
                .bandwidthDeviceToHost(bandwidthDeviceToHost / n)
                .diskReadingTimeNanos(diskReadingTimeNanos / n)
                .bandwidthNanosHostToDevice(bandwidthNanosHostToDevice / n)
                .build();
    }

}
