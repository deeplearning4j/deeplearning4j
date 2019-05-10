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

package org.nd4j.linalg.compression;

import lombok.NonNull;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataTypeEx;

/**
 * This class provides utility methods for Compression in ND4J
 *
 * @author raver119@gmail.com
 */
public class CompressionUtils {

    public static boolean goingToDecompress(@NonNull DataTypeEx from, @NonNull DataTypeEx to) {
        // TODO: eventually we want FLOAT16 here
        if (to.equals(DataTypeEx.FLOAT) || to.equals(DataTypeEx.DOUBLE) )
            return true;

        return false;
    }

    public static boolean goingToCompress(@NonNull DataTypeEx from, @NonNull DataTypeEx to) {
        if (!goingToDecompress(from, to) && goingToDecompress(to, from))
            return true;

        return false;
    }
}
