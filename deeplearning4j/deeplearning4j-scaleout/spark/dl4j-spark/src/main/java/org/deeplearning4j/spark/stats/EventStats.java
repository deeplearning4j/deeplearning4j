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

package org.deeplearning4j.spark.stats;

import java.io.Serializable;

/**
 * Created by Alex on 26/06/2016.
 */
public interface EventStats extends Serializable {

    String getMachineID();

    String getJvmID();

    long getThreadID();

    long getStartTime();

    long getDurationMs();

    /**
     * Get a String representation of the EventStats. This should be a single line delimited representation, suitable
     * for exporting (such as CSV). Should not contain a new-line character at the end of each line
     *
     * @param delimiter Delimiter to use for the data
     * @return String representation of the EventStats object
     */
    String asString(String delimiter);

    /**
     * Get a header line for exporting the EventStats object, for use with {@link #asString(String)}
     *
     * @param delimiter Delimiter to use for the header
     * @return Header line
     */
    String getStringHeader(String delimiter);
}
