/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.optimize.solvers.accumulation;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.shade.guava.util.concurrent.AtomicDouble;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ResidualPostProcessor;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ThresholdAlgorithm;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.ThresholdAlgorithmReducer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.text.DecimalFormat;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This MessageHandler implementation is suited for debugging mostly, but still can be used in production environment if you really want that.
 * Basic idea: updates are encoded before sharing.
 *
 * This handler is used as basement for distributed handler though.
 *
 * PLEASE NOTE: This handler does NOT provide any network connectivity. *
 * @author raver119@gmail.com
 */
@Slf4j
public class EncodingHandler implements MessageHandler {
    public static final long THRESHOLD_LOG_FREQ_MS = 10000; //Every 10 sec max by default
    protected transient GradientsAccumulator accumulator;
    protected ThresholdAlgorithm initialThresholdAlgorithm;
    protected ResidualPostProcessor initialResidualPostProcessor;

    protected Integer boundary;
    protected boolean encodingDebugMode;
    protected AtomicInteger atomicBoundary = new AtomicInteger(-1);

    protected ThreadLocal<ThresholdAlgorithm> thresholdAlgorithm = new ThreadLocal<>();
    protected Map<Long,ThresholdAlgorithm> allThreadThresholdAlgorithms = new ConcurrentHashMap<>();    //All instances - we need to average them at the end once training is complete
    protected ThreadLocal<ResidualPostProcessor> residualPostProcessor = new ThreadLocal<>();
    protected ThreadLocal<AtomicLong> iterations = new ThreadLocal<>();
    protected ThreadLocal<AtomicLong> lastStep = new ThreadLocal<>();
    protected ThreadLocal<AtomicDouble> lastThreshold = new ThreadLocal<>();
    protected ThreadLocal<AtomicDouble> lastSparsityRatio = new ThreadLocal<>();
    protected ThreadLocal<AtomicDouble> currentThreshold = new ThreadLocal<>();
    protected ThreadLocal<AtomicBoolean> bitmapMode = new ThreadLocal<>();
    protected ThreadLocal<AtomicBoolean> lastIterWasDense = new ThreadLocal<>();    //Same as bitmapMode but lagging by 1 iter

    protected final AtomicLong lastThresholdLogTime = new AtomicLong();

    public EncodingHandler(final ThresholdAlgorithm thresholdAlgorithm, final ResidualPostProcessor residualPostProcessor,
                           Integer boundary, boolean encodingDebugMode){
        this.initialThresholdAlgorithm = thresholdAlgorithm;
        this.initialResidualPostProcessor = residualPostProcessor;
        this.boundary = boundary == null ? Integer.MAX_VALUE : boundary;
        this.encodingDebugMode = encodingDebugMode;
    }

    @Override
    public void initialize(@NonNull GradientsAccumulator accumulator) {
        this.accumulator = accumulator;
    }

    public INDArray encodeUpdates(int iteration, int epoch, INDArray updates) {
        if(thresholdAlgorithm.get() == null){
            synchronized (this){
                //Synchronized in case threshold algorithm has INDArrays and we're running on GPU - don't want race condition for shifting devices
                thresholdAlgorithm.set(initialThresholdAlgorithm.clone());
                allThreadThresholdAlgorithms.put(Thread.currentThread().getId(), thresholdAlgorithm.get());
                if(initialResidualPostProcessor != null) {
                    //May be null for no post processing
                    residualPostProcessor.set(initialResidualPostProcessor.clone());
                }
            }
        }

        Double lastThr = null;
        Boolean lastWasDense = null;
        Double lastSparsity = null;
        if(lastThreshold.get() != null){
            //Keep null on first iteration in an epoch, or get for later iterations
            lastThr = lastThreshold.get().get();
            lastWasDense = lastIterWasDense.get().get();
            lastSparsity = lastWasDense || lastSparsityRatio.get() == null ? null : lastSparsityRatio.get().get();
        }



        //Determine current threshold to use:
        double currThreshold = thresholdAlgorithm.get().calculateThreshold(iteration, epoch, lastThr, lastWasDense, lastSparsity, updates);
        if (bitmapMode.get() == null) { //Initialize values for this thread on first iteration (per epoch)
            bitmapMode.set(new AtomicBoolean(true));
            currentThreshold.set(new AtomicDouble(currThreshold));
            iterations.set(new AtomicLong(0));
            lastStep.set(new AtomicLong(0));

            lastThreshold.set(new AtomicDouble(currThreshold));
            lastIterWasDense.set(new AtomicBoolean());
        }

        currentThreshold.get().set(currThreshold);
        lastThreshold.get().set(currThreshold);

        //Debug output if enabled:
        residualDebugOutputIfRequired(updates);

        iterations.get().incrementAndGet();

        if (boundary != null && atomicBoundary.get() < 0)
            atomicBoundary.compareAndSet(-1, (int) (updates.length() / 16) );

        INDArray encoded;

        if (!bitmapMode.get().get()) {
            //Sparse updates
            encoded = Nd4j.getExecutioner().thresholdEncode(updates, currentThreshold.get().get(), boundary == null ? null : atomicBoundary.get());

            // updates were TOO sparse, nothing to share here
            if (encoded == null) {
                bitmapMode.get().set(false);
                if(lastSparsityRatio.get() == null)
                    lastSparsityRatio.set(new AtomicDouble(0.0));
                else
                    lastSparsityRatio.get().set(0.0);
                lastIterWasDense.get().set(false);
                logThresholdIfReq(false, iteration, epoch);
                return null;
            }


            double encLen = encoded.length();

            // if updates are too dense - we fallback to bitmap encoding
            if (encLen >= (updates.length() / 16)) {
                log.debug("Switching back to bitmapEncoding: iteration {}, epoch {}, threshold {}, encoded length {}", iteration, epoch, currThreshold, encLen);
                bitmapMode.get().set(true);

                encoded = Nd4j.getExecutioner().bitmapEncode(updates, currentThreshold.get().get());

                applyPostProcessor(iteration, epoch, currThreshold, updates);
                lastSparsityRatio.set(null);
                lastIterWasDense.get().set(true);
                logThresholdIfReq(true, iteration, epoch);
                return encoded;
            } else {
                //Record sparsity for use in calculation
                double sparsityRatio = encLen / (double)updates.length();
                if(lastSparsityRatio.get() == null){
                    lastSparsityRatio.set(new AtomicDouble(sparsityRatio));
                } else {
                    lastSparsityRatio.get().set(sparsityRatio);
                }
                lastIterWasDense.get().set(false);
            }
        } else {
            //Dense bitmap updates
            encoded = Nd4j.create(DataType.INT32, updates.length() / 16 + 5);

            long values = Nd4j.getExecutioner().bitmapEncode(updates, encoded, currentThreshold.get().get());

            if (values < (updates.length() / 16 + 5) / 2) {
                boolean current = bitmapMode.get().get();
                bitmapMode.get().set(false);
                if(!current) {
                    log.debug("Switched to threshold encoding: iteration {}, epoch {}, threshold {}, number of values {}", iteration, epoch, currThreshold, values);
                }
            }

            lastSparsityRatio.set(null);
            lastIterWasDense.get().set(true);
        }

        //if (encoded != null)
        //log.info("Encoded length: {}, Original/encoded ratio: {}", encoded.data().length(), String.format("%.3f", encoded.data().length() * 100.0 / updates.lengthLong()));
        //log.info("Thread: {}; Encoded length: {}", Thread.currentThread().getId(), Arrays.toString(encoded.data().asInt()));

        applyPostProcessor(iteration, epoch, currThreshold, updates);
        logThresholdIfReq(lastIterWasDense.get().get(), iteration, epoch);
        return encoded;
    }

    public void applyPostProcessor(int iteration, int epoch, Double lastThreshold, INDArray residuals){
        if(initialResidualPostProcessor == null) {
            return; //No op
        }

        residualPostProcessor.get().processResidual(iteration, epoch, lastThreshold, residuals);
    }

    @Deprecated
    public INDArray decodeUpdates(INDArray message) {
        // special op should be called here for decoding

        throw new UnsupportedOperationException();
    }

    /**
     * This method does loops encoded data back to updates queue
     * @param message
     */
    protected void sendMessage(INDArray message, int iterationNumber, int epochNumber) {
        //INDArray update = decodeUpdates(message);
        accumulator.receiveUpdate(message);
    }

    @Override
    public boolean broadcastUpdates(INDArray updates, int iterationNumber, int epochNumber) {
        /*
            we want to do 2 things here:
            1) encode updates
            2) send them somewhere
         */
        INDArray message = encodeUpdates(iterationNumber, epochNumber, updates);
        if (message != null) {
            sendMessage(message, iterationNumber, epochNumber);
            return true;
        } else
            return false;
    }

    protected void logThresholdIfReq(boolean denseUpdates, int iter, int epoch){
        long now = System.currentTimeMillis();
        long lastLog = lastThresholdLogTime.get();
        if(lastLog + THRESHOLD_LOG_FREQ_MS <= now ){
            if (lastThresholdLogTime.compareAndSet(lastLog, now)) { //Avoid RC for logging between multiple threads
                String lastThresholdStr = format(lastThreshold.get().get());
                if(denseUpdates){
                    log.info("Threshold at iter {}, epoch {} [thread {}]: {}, DENSE updates", iter, epoch,
                            Thread.currentThread().getId(), lastThresholdStr);
                } else {
                    AtomicDouble d = lastSparsityRatio.get();
                    String lastSparsityStr;
                    if(d == null)
                        lastSparsityStr = "-";
                    else
                        lastSparsityStr = format(d.get());
                    log.info("Threshold at iter {}, epoch {}: {}, SPARSE updates, last threshold: {}, last sparsity ratio: {}", iter, epoch,
                            Thread.currentThread().getId(), lastThresholdStr, lastSparsityStr);
                }
            }
        }
    }

    protected void residualDebugOutputIfRequired(INDArray residual){
        if(!encodingDebugMode)
            return;

        double currThreshold = currentThreshold.get().get();
        String currThresholdStr = format(currThreshold);


        INDArray absResidual = Transforms.abs(residual, true);

        double dAmean = absResidual.meanNumber().doubleValue();
        double dAMax = absResidual.maxNumber().doubleValue();
        double dPc50 = absResidual.percentileNumber(50).doubleValue();
        double dPc95 = absResidual.percentileNumber(95).doubleValue();
        double dPc99 = absResidual.percentileNumber(99).doubleValue();
        double dPc999 = absResidual.percentileNumber(99.9).doubleValue();
        double dPc9999 = absResidual.percentileNumber(99.99).doubleValue();

        String amean = format(dAmean).replace('E', 'e');
        String aMax = format(dAMax).replace('E', 'e');
        String pc50 = format(dPc50).replace('E', 'e');
        String pc95 = format(dPc95).replace('E', 'e');
        String pc99 = format(dPc99).replace('E', 'e');
        String pc999 = format(dPc999).replace('E', 'e');
        String pc9999 = format(dPc9999).replace('E', 'e');

        String ameanThr = format(dAmean / currThreshold).replace('E', 'e');
        String aMaxThr = format(dAMax / currThreshold).replace('E', 'e');
        String pc50Thr = format(dPc50 / currThreshold).replace('E', 'e');
        String pc95Thr = format(dPc95 / currThreshold).replace('E', 'e');
        String pc99Thr = format(dPc99 / currThreshold).replace('E', 'e');
        String pc999Thr = format(dPc999 / currThreshold).replace('E', 'e');
        String pc9999Thr = format(dPc9999 / currThreshold).replace('E', 'e');

        long length = absResidual.length();
        long countAbsGTEThreshold = absResidual.gte(currThreshold).sumNumber().longValue();
        double sparsity = countAbsGTEThreshold / (double)length;
        String sparsityStr = format(sparsity);

        log.info("Encoding debug info, residual vector: length: {}, threshold: {}, count > thr: {}, sparsity: {}, amean: {} ({}x); amax: {} ({}x); 50%: {} ({}x); 95%: {} ({}x}; 99%: {} ({}x);  99.9%: {} ({}x); 99.99%: {} ({}x)",
                length, currThresholdStr, countAbsGTEThreshold, sparsityStr,
                amean, ameanThr, aMax, aMaxThr, pc50, pc50Thr,
                pc95, pc95Thr, pc99, pc99Thr, pc999, pc999Thr, pc9999, pc9999Thr);
    }

    protected static ThreadLocal<DecimalFormat> formatter = new ThreadLocal<>();
    protected static ThreadLocal<DecimalFormat> formatter2 = new ThreadLocal<>();

    protected static String format(double d){
        if(d == 0){
            return "0.0";
        }
        if((d <= -0.1 && d > -100) ||(d >= 0.1 && d < 100)){
            if(formatter2.get() == null){
                formatter2.set(new DecimalFormat("0.###"));
            }
            return formatter2.get().format(d);
        }

        if(formatter.get() == null){
            formatter.set(new DecimalFormat("0.###E0"));
        }
        DecimalFormat df = formatter.get();
        return df.format(d).replace('E','e');
    }

    /**
     * This should ONLY be called once all training threads have completed
     * @return
     */
    public ThresholdAlgorithm getAverageThresholdAlgorithm(){
        Collection<ThresholdAlgorithm> c = this.allThreadThresholdAlgorithms.values();
        if(c.isEmpty()){
            return null;
        }
        if(c.size() == 1){
            return c.iterator().next();
        }
        Iterator<ThresholdAlgorithm> iter = c.iterator();
        ThresholdAlgorithmReducer r = null;
        while(iter.hasNext()){
            ThresholdAlgorithm ta = iter.next();
            if(r == null){
                r = ta.newReducer();
            }
            r.add(ta);
        }
        ThresholdAlgorithm ta = r.getFinalResult();

        //Remove the old instances in preparation for use in next epoch, if required
        thresholdAlgorithm = new ThreadLocal<>();
        allThreadThresholdAlgorithms.clear();

        return ta;
    }
}
