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

package org.nd4j.parameterserver.distributed.v2.enums;

import org.nd4j.linalg.exception.ND4JIllegalStateException;

/**
 * This enum describes possible result codes for Aeron-powered transmission
 *
 * @author raver119@gmail.com
 */
public enum TransmissionStatus {
    OK,
    NOT_CONNECTED,
    BACK_PRESSURED,
    ADMIN_ACTION,
    CLOSED,
    MAX_POSITION_EXCEEDED;


    public static TransmissionStatus fromLong(long value) {
        if (value == -1)
            return NOT_CONNECTED;
        else if (value == -2)
            return BACK_PRESSURED;
        else if (value == -3)
            return ADMIN_ACTION;
        else if (value == -4)
            return CLOSED;
        else if (value == -5)
            return MAX_POSITION_EXCEEDED;
        else if (value < 0)
            throw new ND4JIllegalStateException("Unknown status returned: [" + value + "]");
        else
            return OK;
    }
}
