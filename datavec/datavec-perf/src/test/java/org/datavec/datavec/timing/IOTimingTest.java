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

package org.datavec.datavec.timing;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.perf.timing.IOTiming;
import org.datavec.perf.timing.TimingStatistics;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.List;

public class IOTimingTest {

    @Test
    public void testTiming() throws Exception  {
        final RecordReader image = new ImageRecordReader(28,28);
        final NativeImageLoader nativeImageLoader = new NativeImageLoader(28,28);

        TimingStatistics timingStatistics = IOTiming.timeNDArrayCreation(image, new ClassPathResource("datavec-perf/largestblobtest.jpg").getInputStream(), new IOTiming.INDArrayCreationFunction() {
            @Override
            public INDArray createFromRecord(List<Writable> record) {
                NDArrayWritable imageWritable = (NDArrayWritable) record.get(0);
                return imageWritable.get();
            }
        });

        System.out.println(timingStatistics);

        TimingStatistics timingStatistics1 = IOTiming.averageFileRead(1000,image,new ClassPathResource("datavec-perf/largestblobtest.jpg").getFile(), new IOTiming.INDArrayCreationFunction() {
            @Override
            public INDArray createFromRecord(List<Writable> record) {
                NDArrayWritable imageWritable = (NDArrayWritable) record.get(0);
                return imageWritable.get();
            }
        });

        System.out.println(timingStatistics1);
    }

}
