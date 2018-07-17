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

package org.datavec.image.recordreader;

import org.apache.commons.io.FileUtils;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.SequenceRecord;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 *
 * A video is just a moving window of pictures.
 * It should be processed as such.
 * This iterates over a root folder and returns a frame
 *
 * @author Adam Gibson
 *
 */
public class VideoRecordReader extends BaseRecordReader implements SequenceRecordReader {
    private Iterator<File> iter;
    private int height = 28, width = 28;
    private BaseImageLoader imageLoader;
    private List<String> labels = new ArrayList<>();
    private boolean appendLabel = false;
    private List<Writable> record;
    private boolean hitImage = false;
    private final List<String> allowedFormats = Arrays.asList("tif", "jpg", "png", "jpeg");
    private Configuration conf;
    public final static String HEIGHT = NAME_SPACE + ".video.height";
    public final static String WIDTH = NAME_SPACE + ".video.width";
    public final static String IMAGE_LOADER = NAME_SPACE + ".imageloader";
    protected InputSplit inputSplit;

    public VideoRecordReader() {}

    /**
     * Load the record reader with the given height and width
     * @param height the height to load
     * @param width the width load
     */
    public VideoRecordReader(int height, int width, List<String> labels) {
        this(height, width, false);
        this.labels = labels;

    }

    public VideoRecordReader(int height, int width, boolean appendLabel, List<String> labels) {
        this.appendLabel = appendLabel;
        this.height = height;
        this.width = width;
        this.labels = labels;

    }

    /**
     * Load the record reader with the given height and width
     * @param height the height to load
     * @param width the width load
     */
    public VideoRecordReader(int height, int width) {
        this(height, width, false);


    }

    public VideoRecordReader(int height, int width, boolean appendLabel) {
        this.appendLabel = appendLabel;
        this.height = height;
        this.width = width;

    }

    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        if (imageLoader == null) {
            imageLoader = new NativeImageLoader(height, width);
        }
        if (split instanceof FileSplit) {
            URI[] locations = split.locations();
            if (locations != null && locations.length >= 1) {
                if (locations.length > 1) {
                    List<File> allFiles = new ArrayList<>();
                    for (URI location : locations) {
                        File iter = new File(location);
                        if (iter.isDirectory()) {
                            allFiles.add(iter);
                            if (appendLabel) {
                                File parentDir = iter.getParentFile();
                                String name = parentDir.getName();
                                if (!labels.contains(name))
                                    labels.add(name);

                            }
                        }

                        else {
                            File parent = iter.getParentFile();
                            if (!allFiles.contains(parent) && containsFormat(iter.getAbsolutePath())) {
                                allFiles.add(parent);
                                if (appendLabel) {
                                    File parentDir = iter.getParentFile();
                                    String name = parentDir.getName();
                                    if (!labels.contains(name))
                                        labels.add(name);

                                }
                            }
                        }

                    }



                    iter = allFiles.iterator();
                } else {
                    File curr = new File(locations[0]);
                    if (!curr.exists())
                        throw new IllegalArgumentException("Path " + curr.getAbsolutePath() + " does not exist!");
                    if (curr.isDirectory())
                        iter = FileUtils.iterateFiles(curr, null, true);
                    else
                        iter = Collections.singletonList(curr).iterator();
                }
            }
        }


        else if (split instanceof InputStreamInputSplit) {
            InputStreamInputSplit split2 = (InputStreamInputSplit) split;
            InputStream is = split2.getIs();
            URI[] locations = split2.locations();
            INDArray load = imageLoader.asMatrix(is);
            record = RecordConverter.toRecord(load);
            if (appendLabel) {
                Path path = Paths.get(locations[0]);
                String parent = path.getParent().toString();
                record.add(new DoubleWritable(labels.indexOf(parent)));
            }

            is.close();
        }



    }

    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        this.conf = conf;
        this.appendLabel = conf.getBoolean(APPEND_LABEL, false);
        this.height = conf.getInt(HEIGHT, height);
        this.width = conf.getInt(WIDTH, width);
        if ("imageio".equals(conf.get(IMAGE_LOADER))) {
            this.imageLoader = new ImageLoader(height, width);
        } else {
            this.imageLoader = new NativeImageLoader(height, width);
        }

        initialize(split);
    }

    private boolean containsFormat(String format) {
        for (String format2 : allowedFormats)
            if (format.endsWith("." + format2))
                return true;
        return false;
    }

    @Override
    public List<Writable> next() {
        if (iter != null) {
            List<Writable> ret = new ArrayList<>();
            File image = iter.next();
            if (image.isDirectory() || !containsFormat(image.getAbsolutePath()))
                return next();
            try {
                invokeListeners(image);
                INDArray row = imageLoader.asRowVector(image);
                for (int i = 0; i < row.length(); i++)
                    ret.add(new DoubleWritable(row.getDouble(i)));
                if (appendLabel)
                    ret.add(new DoubleWritable(labels.indexOf(image.getParentFile().getName())));
            } catch (Exception e) {
                e.printStackTrace();
            }
            if (iter.hasNext()) {
                return ret;
            } else {
                if (iter.hasNext()) {
                    try {
                        ret.add(new Text(FileUtils.readFileToString(iter.next())));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

            }

            return ret;
        }

        else if (record != null) {
            hitImage = true;
            invokeListeners(record);
            return record;
        }


        throw new IllegalStateException("No more elements");

    }

    @Override
    public boolean hasNext() {
        if (iter != null) {
            return iter.hasNext();
        } else if (record != null) {
            return !hitImage;
        }
        throw new IllegalStateException("Indeterminant state: record must not be null, or a file iterator must exist");
    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public List<List<Writable>> sequenceRecord() {
        File next = iter.next();
        invokeListeners(next);
        if (!next.isDirectory())
            return Collections.emptyList();
        File[] list = next.listFiles();
        List<List<Writable>> ret = new ArrayList<>();
        for (File f : list) {
            try {
                List<Writable> record = RecordConverter.toRecord(imageLoader.asRowVector(f));
                ret.add(record);
                if (appendLabel)
                    record.add(new DoubleWritable(labels.indexOf(next.getName())));
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

        }
        return ret;
    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }


    @Override
    public void reset() {
        if (inputSplit == null)
            throw new UnsupportedOperationException("Cannot reset without first initializing");
        try {
            initialize(inputSplit);
        } catch (Exception e) {
            throw new RuntimeException("Error during VideoRecordReader reset", e);
        }
    }

    @Override
    public boolean resetSupported(){
        if(inputSplit == null){
            return false;
        }
        return inputSplit.resetSupported();
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException(
                        "Loading video data via VideoRecordReader + DataInputStream not supported.");
    }

    @Override
    public Record nextRecord() {
        throw new UnsupportedOperationException("Use nextSequence() for sequence readers");
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        throw new UnsupportedOperationException("Use loadSequenceFromMeta() for sequence readers");
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        throw new UnsupportedOperationException("Loading from metadata not yet implemented");
    }

    @Override
    public List<List<Writable>> sequenceRecord(URI uri, DataInputStream dataInputStream) throws IOException {
        throw new UnsupportedOperationException(
                        "Loading video data via VideoRecordReader + DataInputStream not supported.");
    }

    @Override
    public SequenceRecord nextSequence() {
        return new org.datavec.api.records.impl.SequenceRecord(sequenceRecord(), null);
    }

    @Override
    public SequenceRecord loadSequenceFromMetaData(RecordMetaData recordMetaData) throws IOException {
        throw new UnsupportedOperationException("Loading from metadata not yet implemented");
    }

    @Override
    public List<SequenceRecord> loadSequenceFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        throw new UnsupportedOperationException("Loading from metadata not yet implemented");
    }

}
