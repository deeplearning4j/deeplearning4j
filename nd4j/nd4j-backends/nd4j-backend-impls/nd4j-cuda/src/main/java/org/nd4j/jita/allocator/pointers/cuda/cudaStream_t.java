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

package org.nd4j.jita.allocator.pointers.cuda;

import lombok.NonNull;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.linalg.exception.ND4JException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

/**
 * @author raver119@gmail.com
 */
public class cudaStream_t extends CudaPointer {

    public cudaStream_t(@NonNull Pointer pointer) {
        super(pointer);
    }

    public int synchronize() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int res = nativeOps.streamSynchronize(this);
        if (res == 0)
            throw new ND4JException("CUDA exception happened. Terminating. Last op: [" + Nd4j.getExecutioner().getLastOp() +"]");

        return res;
    }
}
