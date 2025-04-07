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

package org.deeplearning4j.nn.layers.samediff;

import org.nd4j.autodiff.samediff.internal.memory.AbstractMemoryMgr;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

public class DL4JSameDiffMemoryMgr extends AbstractMemoryMgr {

    private final String workingMemoryWs;
    private final String outputWs;
    private final WorkspaceConfiguration confWorking;
    private final WorkspaceConfiguration confOutput;

    //Note: if the working memory or output workspace names are null -> detached memory
    public DL4JSameDiffMemoryMgr(String workingMemoryWs, String outputWs, WorkspaceConfiguration confWorking,
                                 WorkspaceConfiguration confOutput) {
        this.workingMemoryWs = workingMemoryWs;
        this.outputWs = outputWs;
        this.confWorking = confWorking;
        this.confOutput = confOutput;
    }


    @Override
    public INDArray allocate(boolean detached, DataType dataType, long... shape) {
        String wsName = detached ? outputWs : workingMemoryWs;
        WorkspaceConfiguration wsConf = detached ? confOutput : confWorking;

        if(wsName == null) {
            //Scoped out
            INDArray ret = Nd4j.createUninitializedDetached(dataType, shape);
            Preconditions.checkState(!ret.isAttached(), "Returned array should be detached");
            return ret;
        } else {
            MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(wsConf, wsName);
            ws.notifyScopeBorrowed();
            return Nd4j.createUninitialized(dataType, shape);

        }
    }

    @Override
    public INDArray allocate(boolean detached, LongShapeDescriptor descriptor) {
        if(descriptor.isEmpty()) {
            INDArray ret =  Nd4j.create(descriptor);
            if(detached) {
                ret = ret.detach();
            }

            return ret;
        }

        return allocate(detached, descriptor.dataType(), descriptor.getShape());
    }

    @Override
    public void release(INDArray array) {
        //No-op - DL4J workspaces handles this
    }

    @Override
    public void close() {
        //No-op - DL4J workspaces handles this
    }

    @Override
    public INDArray allocateFromDescriptor(boolean detached, DataBuffer dataBuffer) {
        long[] shapeInfo = dataBuffer.asLong();
        DataType dataType = Shape.dataType(shapeInfo);
        long[] shape = Shape.shape(shapeInfo);
        String wsName = detached ? outputWs : workingMemoryWs;
        WorkspaceConfiguration wsConf = detached ? confOutput : confWorking;

        if(wsName == null) {
            //Scoped out
            INDArray ret = Nd4j.createUninitializedDetached(dataType, shape);
            Preconditions.checkState(!ret.isAttached(), "Returned array should be detached");
            return ret;
        } else {
            MemoryWorkspace ws = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(wsConf, wsName);
            ws.notifyScopeBorrowed();
            return Nd4j.createUninitialized(dataType, shape);

        }

    }
}
