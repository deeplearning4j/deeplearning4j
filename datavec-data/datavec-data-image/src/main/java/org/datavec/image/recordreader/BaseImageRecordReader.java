/*
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.image.recordreader;

import org.apache.commons.io.FileUtils;
import org.datavec.api.conf.Configuration;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.records.reader.RecordReaderMeta;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.common.RecordConverter;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;
import java.net.URI;
import java.util.*;

/**
 * Base class for the image record reader
 *
 * @author Adam Gibson
 */
public abstract class BaseImageRecordReader extends BaseRecordReader implements RecordReaderMeta {
    protected Iterator<File> iter;
    protected Configuration conf;
    protected File currentFile;
    protected PathLabelGenerator labelGenerator = null;
    protected List<String> labels = new ArrayList<>();
    protected boolean appendLabel = false;
    protected List<Writable> record;
    protected boolean hitImage = false;
    protected int height = 28, width = 28, channels = 1;
    protected boolean cropImage = false;
    protected ImageTransform imageTransform;
    protected BaseImageLoader imageLoader;
    protected InputSplit inputSplit;
    protected Map<String, String> fileNameMap = new LinkedHashMap<>();
    protected String pattern; // Pattern to split and segment file name, pass in regex
    protected int patternPosition = 0;
    protected double normalizeValue = 0;

    public final static String HEIGHT = NAME_SPACE + ".height";
    public final static String WIDTH = NAME_SPACE + ".width";
    public final static String CHANNELS = NAME_SPACE + ".channels";
    public final static String CROP_IMAGE = NAME_SPACE + ".cropimage";
    public final static String IMAGE_LOADER = NAME_SPACE + ".imageloader";

    public BaseImageRecordReader() {
    }

    public BaseImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator) {
        this(height, width, channels,labelGenerator, null, 0.0);
    }

    public BaseImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator, ImageTransform imageTransform, double normalizeValue) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.labelGenerator = labelGenerator;
        this.imageTransform = imageTransform;
        this.appendLabel = labelGenerator !=null? true: false;
        this.normalizeValue = normalizeValue;
    }

    protected boolean containsFormat(String format) {
        for (String format2 : imageLoader.getAllowedFormats())
            if (format.endsWith("." + format2))
                return true;
        return false;
    }


    @Override
    public void initialize(InputSplit split) throws IOException {
        if (imageLoader == null) {
            imageLoader = new NativeImageLoader(height, width, channels, imageTransform, normalizeValue);
        }
        inputSplit = split;
        Collection<File> allFiles;
        URI[] locations = split.locations();
        if (locations != null && locations.length >= 1) {
            if (locations.length > 1 || containsFormat(locations[0].getPath())) {
                allFiles = new ArrayList<>();
                for (URI location : locations) {
                    File imgFile = new File(location);
                    if (!imgFile.isDirectory() && containsFormat(imgFile.getAbsolutePath()))
                        allFiles.add(imgFile);
                    if (appendLabel) {
                        File parentDir = imgFile.getParentFile();
                        String name = parentDir.getName();
                        if (labelGenerator != null) {
                            name = labelGenerator.getLabelForPath(location).toString();
                        }
                        if (!labels.contains(name))
                            labels.add(name);
                        if (pattern != null) {
                            String label = name.split(pattern)[patternPosition];
                            fileNameMap.put(imgFile.toString(), label);
                        }
                    }
                }
            } else {
                File curr = new File(locations[0]);
                if (!curr.exists())
                    throw new IllegalArgumentException("Path " + curr.getAbsolutePath() + " does not exist!");
                if (curr.isDirectory()) {
                    allFiles = FileUtils.listFiles(curr, null, true);
                } else {
                    allFiles = Collections.singletonList(curr);
                }

            }
            iter = allFiles.iterator();
        }
        if (split instanceof FileSplit) {
            //remove the root directory
            FileSplit split1 = (FileSplit) split;
            labels.remove(split1.getRootDir());
        }
    }


    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        this.appendLabel = conf.getBoolean(APPEND_LABEL, false);
        this.labels = new ArrayList<>(conf.getStringCollection(LABELS));
        this.height = conf.getInt(HEIGHT, height);
        this.width = conf.getInt(WIDTH, width);
        this.channels = conf.getInt(CHANNELS, channels);
        this.cropImage = conf.getBoolean(CROP_IMAGE, cropImage);
        if ("imageio".equals(conf.get(IMAGE_LOADER))) {
            this.imageLoader = new ImageLoader(height, width, channels, cropImage);
        } else {
            this.imageLoader = new NativeImageLoader(height, width, channels, imageTransform, normalizeValue);
        }
        this.conf = conf;
        initialize(split);
    }


    /**
     * Called once at initialization.
     *
     * @param split          the split that defines the range of records to read
     * @param imageTransform the image transform to use to transform images while loading them
     * @throws java.io.IOException
     * @throws InterruptedException
     */
    public void initialize(InputSplit split, ImageTransform imageTransform) throws IOException {
        this.imageLoader = null;
        this.imageTransform = imageTransform;
        initialize(split);
    }

    /**
     * Called once at initialization.
     *
     * @param conf           a configuration for initialization
     * @param split          the split that defines the range of records to read
     * @param imageTransform the image transform to use to transform images while loading them
     * @throws java.io.IOException
     * @throws InterruptedException
     */
    public void initialize(Configuration conf, InputSplit split, ImageTransform imageTransform) throws IOException, InterruptedException {
        this.imageLoader = null;
        this.imageTransform = imageTransform;
        initialize(conf, split);
    }


    @Override
    public List<Writable> next() {
        if (iter != null) {
            List<Writable> ret = new ArrayList<>();
            File image =  iter.next();
            currentFile = image;

            if (image.isDirectory())
                return next();
            try {
                invokeListeners(image);
                INDArray row = imageLoader.asMatrix(image);
                ret = RecordConverter.toRecord(row);
                if (appendLabel)
                    ret.add(new IntWritable(labels.indexOf(getLabel(image.getPath()))));
            } catch (Exception e) {
                e.printStackTrace();
            }
            return ret;
        } else if (record != null) {
            hitImage = true;
            invokeListeners(record);
            return record;
        }
        throw new IllegalStateException("No more elements");
    }

    @Override
    public boolean hasNext() {
        if (iter != null) {
            boolean hasNext = iter.hasNext();
            if (!hasNext && imageTransform != null) {
                imageTransform.transform(null);
            }
            return hasNext;
        } else if (record != null) {
            if (hitImage && imageTransform != null) {
                imageTransform.transform(null);
            }
            return !hitImage;
        }
        if (imageTransform != null) {
            imageTransform.transform(null);
        }
        throw new IllegalStateException("Indeterminant state: record must not be null, or a file iterator must exist");
    }

    @Override
    public void close() throws IOException {

    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }


    /**
     * Get the label from the given path
     *
     * @param path the path to get the label from
     * @return the label for the given path
     */
    public String getLabel(String path) {
        if (labelGenerator != null) {
            return labelGenerator.getLabelForPath(path).toString();
        }
        if (fileNameMap != null && fileNameMap.containsKey(path)) return fileNameMap.get(path);
        return (new File(path)).getParentFile().getName();
    }

    /**
     * Accumulate the label from the path
     *
     * @param path the path to get the label from
     */
    protected void accumulateLabel(String path) {
        String name = getLabel(path);
        if (!labels.contains(name))
            labels.add(name);
    }

    /**
     * Returns the file loaded last by {@link #next()}.
     */
    public File getCurrentFile() {
        return currentFile;
    }

    /**
     * Sets manually the file returned by {@link #getCurrentFile()}.
     */
    public void setCurrentFile(File currentFile) {
        this.currentFile = currentFile;
    }

    @Override
    public List<String> getLabels() {
        return labels;
    }

    public void setLabels(List<String> labels) {
        this.labels = labels;
    }

    @Override
    public void reset() {
        if (inputSplit == null) throw new UnsupportedOperationException("Cannot reset without first initializing");
        try {
            initialize(inputSplit);
        } catch (Exception e) {
            throw new RuntimeException("Error during LineRecordReader reset", e);
        }
    }

    /**
     * Returns {@code getLabels().size()}.
     */
    public int numLabels() {
        return labels.size();
    }

    @Override
    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        invokeListeners(uri);
        if (imageLoader == null) {
            imageLoader = new NativeImageLoader(height, width, channels, imageTransform);
        }
        INDArray row = imageLoader.asMatrix(dataInputStream);
        List<Writable> ret = RecordConverter.toRecord(row);
        if (appendLabel) ret.add(new IntWritable(labels.indexOf(getLabel(uri.getPath()))));
        return ret;
    }

    @Override
    public Record nextRecord() {
        List<Writable> list = next();
        URI uri = currentFile.toURI();
        return new org.datavec.api.records.impl.Record(list, new RecordMetaDataURI(uri, BaseImageRecordReader.class));
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<Record> out = new ArrayList<>();
        for(RecordMetaData meta : recordMetaDatas){
            URI uri = meta.getURI();
            File f = new File(uri);

            List<Writable> next;
            try(DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(f)))){
                next = record(uri, dis);
            }
            out.add(new org.datavec.api.records.impl.Record(next, meta));
        }
        return out;
    }
}
