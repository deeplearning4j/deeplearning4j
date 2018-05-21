package org.datavec.datavec.timing;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.util.ClassPathResource;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.perf.timing.IOTiming;
import org.datavec.perf.timing.TimingStatistics;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

public class IOTimingTest {

    @Test
    public void testTiming() throws Exception  {
        final RecordReader image = new ImageRecordReader(28,28);
        final NativeImageLoader nativeImageLoader = new NativeImageLoader(28,28);
        TimingStatistics timingStatistics = IOTiming.timeNDArrayCreation(image, new ClassPathResource("largestblobtest.jpg").getInputStream(), new IOTiming.INDArrayCreationFunction() {
            @Override
            public INDArray createFromRecord(List<Writable> record) {
                NDArrayWritable imageWritable = (NDArrayWritable) record.get(0);
                return imageWritable.get();
            }
        });

        System.out.println(timingStatistics);

        TimingStatistics timingStatistics1 = IOTiming.averageFileRead(1000,image,new ClassPathResource("largestblobtest.jpg").getFile(), new IOTiming.INDArrayCreationFunction() {
            @Override
            public INDArray createFromRecord(List<Writable> record) {
                NDArrayWritable imageWritable = (NDArrayWritable) record.get(0);
                return imageWritable.get();
            }
        });

        System.out.println(timingStatistics1);
    }

}
