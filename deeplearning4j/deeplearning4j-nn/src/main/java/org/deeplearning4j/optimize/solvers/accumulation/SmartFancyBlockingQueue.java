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

package org.deeplearning4j.optimize.solvers.accumulation;


import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.ThresholdCompression;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.AtomicBoolean;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This class provides additional functionality to FancyBlockingQueue: it tracks memory use of stored compressed INDArrays, and if their size becomes too big, it:
 * a) decompresses them into single INDArray
 * b) removes original updates messages
 * c) keeps updating single INDArray until it gets consumed
 * d) once that happened - it automatically switches back to original behavior
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class SmartFancyBlockingQueue extends FancyBlockingQueue<INDArray> {
    protected final ReentrantReadWriteLock smartLock = new ReentrantReadWriteLock();
    protected int decompressionThreshold = 32;
    protected AtomicBoolean collapsedMode = new AtomicBoolean(false);


    protected final long[] paramsShape;
    protected final char paramsOrder;

    public SmartFancyBlockingQueue(int decompressionThreshold, @NonNull INDArray paramsMatrix) {
        this(decompressionThreshold, new LinkedBlockingQueue<INDArray>(1024), paramsMatrix);
    }

    public SmartFancyBlockingQueue(int decompressionThreshold, BlockingQueue<INDArray> queue, @NonNull INDArray paramsMatrix) {
        super(queue);
        this.decompressionThreshold = decompressionThreshold;

        this.paramsShape = paramsMatrix.shape();
        this.paramsOrder = paramsMatrix.ordering();
    }

    protected INDArray smartDecompress(INDArray encoded, INDArray target) {
        INDArray result = target == null ? Nd4j.create(paramsShape, paramsOrder) : target;

        if (encoded.isCompressed() || encoded.data().dataType() == DataBuffer.Type.INT) {
            int encoding = encoded.data().getInt(3);
            if (encoding == ThresholdCompression.FLEXIBLE_ENCODING) {
                Nd4j.getExecutioner().thresholdDecode(encoded, result);
            } else if (encoding == ThresholdCompression.BITMAP_ENCODING) {
                Nd4j.getExecutioner().bitmapDecode(encoded, result);
            } else
                throw new ND4JIllegalStateException("Unknown encoding mode: [" + encoding + "]");
        } else {
            result.addi(encoded);
        }

        return result;
    }

    @Override
    public void put(INDArray array) throws InterruptedException {
        try {
            smartLock.writeLock().lock();

            if (backingQueue.size() > decompressionThreshold || collapsedMode.get()) {
                collapsedMode.set(true);

                log.info("Collapsing updates...");

                // if we're already in collapsed mode - we'll just poll back our single collapsed array and update it
                INDArray params = smartDecompress(array, backingQueue.size() == 1 ? backingQueue.poll() : null);
                while (!backingQueue.isEmpty()) {
                    val arr = backingQueue.poll();
                    smartDecompress(arr, params);
                }

                // now just put single array back
                super.put(params);
            } else
                super.put(array);
        } finally {
            smartLock.writeLock().unlock();
        }
    }

    @Override
    public INDArray poll() {
        try {
            // we use this lock to make
            smartLock.readLock().lock();

            // from now on this SFBQ instance won't add up to single compressed array
            collapsedMode.set(false);

            return super.poll();
        } finally {
            smartLock.readLock().unlock();
        }
    }
}
