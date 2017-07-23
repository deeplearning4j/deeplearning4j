package org.deeplearning4j.optimize.solvers.accumulation;

import com.google.common.util.concurrent.AtomicDouble;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.compression.NDArrayCompressor;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;
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
    protected transient GradientsAccumulator accumulator;
    protected double threshold, minThreshold, thresholdStep;
    protected int shakeFrequency;
    protected Double boundary = null;
    protected NDArrayCompressor compressor;
    protected AtomicInteger atomicBoundary = new AtomicInteger(-1);

    protected ThreadLocal<AtomicLong> iterations = new ThreadLocal<>();
    protected ThreadLocal<AtomicLong> lastStep = new ThreadLocal<>();
    protected ThreadLocal<AtomicDouble> currentThreshold = new ThreadLocal<>();
    protected ThreadLocal<AtomicBoolean> bitmapMode = new ThreadLocal<>();

    public EncodingHandler() {
        this(1e-3);
    }

    public EncodingHandler(double threshold) {
        this(threshold, null);
    }

    public EncodingHandler(double threshold, Double boundary) {
        this(threshold, threshold, 0.0, 0, boundary);
    }

    public EncodingHandler(double threshold, double minThreshold, double thresholdStep,  int shakeFrequency) {
        this(threshold, minThreshold, thresholdStep, shakeFrequency, null);
    }

    public EncodingHandler(double threshold, double minThreshold, double thresholdStep, int shakeFrequency, Double boundary) {
        this.threshold = threshold;
        this.minThreshold = minThreshold;
        this.thresholdStep= thresholdStep;
        this.shakeFrequency = shakeFrequency;
        this.boundary = boundary;
    }

    @Override
    public void initialize(@NonNull GradientsAccumulator accumulator) {
        this.accumulator = accumulator;

        compressor = Nd4j.getCompressor().getCompressor("THRESHOLD");
        if (compressor == null)
            throw new ND4JIllegalStateException("Can't find Threshold compressor implementation!");

        compressor.configure(threshold);
    }

    public INDArray encodeUpdates(INDArray updates) {
        // special op should be called here for encoding
        if (bitmapMode.get() == null) {
            bitmapMode.set(new AtomicBoolean(true));
            currentThreshold.set(new AtomicDouble(threshold));
            iterations.set(new AtomicLong(0));
            lastStep.set(new AtomicLong(0));
        }

        iterations.get().incrementAndGet();

        if (boundary != null && atomicBoundary.get() < 0)
            atomicBoundary.compareAndSet(-1, (int) (updates.lengthLong() * boundary));

        INDArray encoded =  null;

        if (!bitmapMode.get().get()) {
            // if shakeFrequency hits here, we'll use bitmap encoding for one round for 1/2 of current threshold
            if (shakeFrequency != 0 && iterations.get().get() % shakeFrequency == 0) {
                DataBuffer buffer = Nd4j.getDataBufferFactory().createInt(updates.lengthLong() / 16 + 5);
                encoded = Nd4j.createArrayFromShapeBuffer(buffer, updates.shapeInfoDataBuffer());

                Nd4j.getExecutioner().bitmapEncode(updates, encoded, currentThreshold.get().get() / 2);
            } else {
                // otherwise (probably most often - we go for sparse
                encoded = Nd4j.getExecutioner().thresholdEncode(updates, currentThreshold.get().get(), boundary == null ? null : atomicBoundary.get());
                double encLen = encoded.data().getInt(0);
                double encodingRatio = encLen * 100.0 / updates.length();

                // if updates are too dense - we fallback to bitmap encoding
                if (encLen > (updates.lengthLong() / 16 + 5))
                    bitmapMode.get().set(true);


                // after encoding is finished, and updates are sparse enough - let's step down a bit
                // and we don't step down too early, so we wait for 50 iterations at least to step down
                if (minThreshold <= currentThreshold.get().get() && minThreshold < currentThreshold.get().get() - thresholdStep && iterations.get().get() > lastStep.get().get() + 50 && encodingRatio < 0.05) {
                    currentThreshold.get().addAndGet(-thresholdStep);
                    lastStep.set(iterations.get());
                    log.info("Threshold steps down to {}", currentThreshold.get().get());
                }
            }
        } else {
            DataBuffer buffer = Nd4j.getDataBufferFactory().createInt(updates.lengthLong() / 16 + 5);
            encoded = Nd4j.createArrayFromShapeBuffer(buffer, updates.shapeInfoDataBuffer());

            long values = Nd4j.getExecutioner().bitmapEncode(updates, encoded, currentThreshold.get().get());

            if (values < (updates.lengthLong() / 16 + 5) / 2) {
                bitmapMode.get().set(false);
                log.info("Switched to threshold encoding");
            }
        }

        //if (encoded != null)
        //log.info("Encoded length: {}, Original/encoded ratio: {}", encoded.data().length(), String.format("%.3f", encoded.data().length() * 100.0 / updates.lengthLong()));
        //log.info("Thread: {}; Encoded length: {}", Thread.currentThread().getId(), Arrays.toString(encoded.data().asInt()));

        return encoded;
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
    protected void sendMessage(INDArray message) {
        //INDArray update = decodeUpdates(message);
        accumulator.receiveUpdate(message);
    }

    @Override
    public boolean broadcastUpdates(INDArray updates) {
        /*
            we want to do 2 things here:
            1) encode updates
            2) send them somewhere
         */
        INDArray message = encodeUpdates(updates);
        if (message != null) {
            sendMessage(message);
            return true;
        } else
            return false;
    }
}
