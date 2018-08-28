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

package org.deeplearning4j.spark.parameterserver.networking.v2;

import io.reactivex.functions.Consumer;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.solvers.accumulation.FancyBlockingQueue;
import org.deeplearning4j.optimize.solvers.accumulation.GradientsAccumulator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.ThresholdCompression;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.reactivestreams.Subscriber;
import org.reactivestreams.Subscription;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 *  This Subscriber is responsible for gradient updates application
 *
 * @author raver119@gmail.com
 */
@AllArgsConstructor
@NoArgsConstructor
@Builder
@Slf4j
public class UpdatesConsumer implements Subscriber<INDArray> {
    protected transient INDArray params;
    protected transient INDArray updates;
    protected transient StepFunction stepFunction;

    protected transient GradientsAccumulator accumulator;

    protected transient final AtomicLong updatesCount = new AtomicLong(0);
    protected transient final AtomicBoolean hasSomething = new AtomicBoolean(false);
    protected transient final AtomicBoolean bypassMode = new AtomicBoolean(false);
    protected transient final AtomicLong denseCounter = new AtomicLong(0);
    protected transient final AtomicLong sparseCounter = new AtomicLong(0);

    // make this stuff configurable
    protected transient final BlockingQueue<INDArray> updatesBuffer = new FancyBlockingQueue<>(new LinkedBlockingQueue<>(1024));

    @Override
    public void onSubscribe(Subscription subscription) {
        // no-op
    }

    /**
     * This
     * @param reallBypass
     */
    public void bypassMode(boolean reallBypass) {
        bypassMode.set(reallBypass);
    }

    /**
     *
     * @return
     */
    public boolean isBypassMod() {
        return bypassMode.get();
    }

    @Override
    public void onNext(INDArray array) {
        if (!bypassMode.get()) {
            if (accumulator != null) {
                // this means consumer runs on worker node

                try {
                    // we're just storing update into buffer, and it'll be consumed by GradientsAccumulator on next cycle
                    updatesBuffer.put(array);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            } else if (params != null && stepFunction != null) {
                synchronized (this) {
                    // threshold decoder is inplace & fast
                    int encoding = array.data().getInt(3);
                    if (encoding == ThresholdCompression.FLEXIBLE_ENCODING) {
                        Nd4j.getExecutioner().thresholdDecode(array, updates);
                        sparseCounter.incrementAndGet();
                    } else if (encoding == ThresholdCompression.BITMAP_ENCODING) {
                        Nd4j.getExecutioner().bitmapDecode(array, updates);
                        denseCounter.incrementAndGet();
                    } else
                        throw new DL4JInvalidConfigException("Unknown compression header received: " + encoding);


                    // this simple flag shows that we have something not applied, will be used at finishTraining() method
                    hasSomething.set(true);

                    // we apply updates every X iterations, and we don't really need X to be small here
                    if (updatesCount.incrementAndGet() > 0) {
                        stepFunction.step(params, updates);
                        Nd4j.getExecutioner().commit();

                        log.info("Applying updates. Current ratio: [{}]", (double) sparseCounter.get() / denseCounter.get());

                        // once accumulated updates are applied - reset storage, and wait for other messsages
                        Nd4j.getMemoryManager().memset(updates);
                        hasSomething.set(false);
                    }
                }
            } else
                throw new ND4JIllegalStateException("Accumulator & StepFunction is null at the same time");
        }
    }

    @Override
    public void onError(Throwable throwable) {
        throw new RuntimeException(throwable);
    }

    @Override
    public void onComplete() {
        // no-op
    }
}
