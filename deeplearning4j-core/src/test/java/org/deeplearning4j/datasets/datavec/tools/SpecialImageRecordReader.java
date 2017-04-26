package org.deeplearning4j.datasets.datavec.tools;

import org.apache.commons.lang3.RandomUtils;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.common.RecordConverter;
import org.datavec.common.data.NDArrayWritable;
import org.datavec.image.recordreader.ImageRecordReader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 *
 * @author raver119@gmail.com
 */
public class SpecialImageRecordReader extends ImageRecordReader {
    private AtomicInteger counter = new AtomicInteger(0);
    private int limit, channels, width, height, numClasses;
    private List<String> labels = new ArrayList<>();



    public SpecialImageRecordReader(int totalExamples, int numClasses, int channels, int width, int height) {
        this.limit = totalExamples;
        this.channels = channels;
        this.width = width;
        this.height = height;
        this.numClasses = numClasses;

        for (int i = 0; i < numClasses; i++) {
            labels.add("" + i);
        }
    }

    @Override
    public boolean hasNext() {
        return counter.get() < limit;
    }


    @Override
    public void reset() {
        counter.set(0);
    }

    @Override
    public List<Writable> next() {
        INDArray features = Nd4j.create(1, channels, height, width).assign(counter.getAndIncrement());
        List<Writable> ret = RecordConverter.toRecord(features);
        ret.add(new IntWritable(RandomUtils.nextInt(0, numClasses)));
        return ret;
    }

    public List<String> getLabels() {
        return labels;
    }


    @Override
    public boolean batchesSupported() {
        return true;
    }

    @Override
    public List<Writable> next(int num) {
        int numExamples = Math.min(num, limit - counter.get());
        //counter.addAndGet(numExamples);

        INDArray features = Nd4j.create(numExamples, channels, height, width);
        for (int i = 0; i < numExamples; i++) {
            features.tensorAlongDimension(i, 1, 2, 3).assign(counter.getAndIncrement());
        }

        INDArray labels = Nd4j.create(numExamples, numClasses);

        List<Writable> ret = RecordConverter.toRecord(features);
        ret.add(new NDArrayWritable(labels));

        return ret;
    }
}
