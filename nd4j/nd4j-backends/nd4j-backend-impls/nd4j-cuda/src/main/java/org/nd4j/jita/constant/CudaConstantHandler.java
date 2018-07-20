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

package org.nd4j.jita.constant;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.cache.BasicConstantHandler;
import org.nd4j.linalg.cache.ConstantHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ConstantHandler implementation for CUDA backend.
 *
 * @author raver119@gmail.com
 */
public class CudaConstantHandler extends BasicConstantHandler {
    private static Logger logger = LoggerFactory.getLogger(CudaConstantHandler.class);

    protected static final ConstantHandler wrappedHandler = ProtectedCudaConstantHandler.getInstance();

    public CudaConstantHandler() {

    }

    @Override
    public long moveToConstantSpace(DataBuffer dataBuffer) {
        return wrappedHandler.moveToConstantSpace(dataBuffer);
    }

    @Override
    public DataBuffer getConstantBuffer(int[] array) {
        return wrappedHandler.getConstantBuffer(array);
    }

    @Override
    public DataBuffer getConstantBuffer(float[] array) {
        return wrappedHandler.getConstantBuffer(array);
    }

    @Override
    public DataBuffer getConstantBuffer(double[] array) {
        return wrappedHandler.getConstantBuffer(array);
    }

    @Override
    public DataBuffer getConstantBuffer(long[] array) {
        return wrappedHandler.getConstantBuffer(array);
    }

    @Override
    public DataBuffer relocateConstantSpace(DataBuffer dataBuffer) {
        return wrappedHandler.relocateConstantSpace(dataBuffer);
    }

    /**
     * This method removes all cached constants
     */
    @Override
    public void purgeConstants() {
        wrappedHandler.purgeConstants();
    }

    @Override
    public long getCachedBytes() {
        return wrappedHandler.getCachedBytes();
    }
}
