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

package org.nd4j.autodiff.samediff.internal.memory;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;

public class NoOpMemoryMgr extends AbstractMemoryMgr implements SessionMemMgr {

    @Override
    public INDArray allocate(boolean detached, DataType dataType, long... shape) {
        return Nd4j.createUninitialized(dataType, shape);
    }

    @Override
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor) {
        return Nd4j.create(descriptor, false);
    }

    @Override
    public void release(@NonNull INDArray array) {
        //No-op, rely on GC to clear arrays
    }

    @Override
    public void close() {
        //No-op
    }

    @Override
    public INDArray allocateFromDescriptor(boolean detached, DataBuffer dataBuffer) {
       return Nd4j.createFromDescriptor(dataBuffer);
    }

}