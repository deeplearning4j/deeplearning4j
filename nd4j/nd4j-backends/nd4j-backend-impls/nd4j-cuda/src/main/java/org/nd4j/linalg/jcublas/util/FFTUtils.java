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

package org.nd4j.linalg.jcublas.util;

import org.nd4j.linalg.api.buffer.DataBuffer;

/**
 * @author Adam Gibson
 */
public class FFTUtils {
    /**
     * Get the plan for the given buffer (C2C for float Z2Z for double)
     * @param buff the buffer to get the plan for
     * @return the plan for the given buffer
     */
    public static int getPlanFor(DataBuffer buff) {
        /*   if(buff.dataType() == DataBuffer.Type.FLOAT)
            return cufftType.CUFFT_C2C;
        else
            return cufftType.CUFFT_Z2Z;
            */
        throw new UnsupportedOperationException();
    }


}
