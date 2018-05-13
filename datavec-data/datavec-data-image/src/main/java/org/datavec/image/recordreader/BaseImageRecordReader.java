/*-
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

import com.google.common.base.Preconditions;
import org.datavec.api.conf.Configuration;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.io.labels.PathMultiLabelGenerator;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataURI;
import org.datavec.api.records.reader.BaseRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.InputStreamInputSplit;
import org.datavec.api.util.files.FileFromPathIterator;
import org.datavec.api.util.files.URIUtil;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.batch.NDArrayRecordBatch;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.net.URI;
import java.util.*;

/**
 * Base class for the image record reader
 *
 * @author Adam Gibson
 */
public abstract class BaseImageRecordReader extends BaseRecordReader {
    protected boolean finishedInputStreamSplit;
    protected Iterator<File> iter;
    protected Configuration conf;
    protected File currentFile;
    protected PathLabelGenerator labelGenerator = null;
    protected PathMultiLabelGenerator labelMultiGenerator = null;
    protected List<String> labels = new ArrayList<>();
    protected boolean appendLabel = false;
    protected boolean writeLabel = false;
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

    public final static String HEIGHT = NAME_SPACE + ".height";
    public final static String WIDTH = NAME_SPACE + ".width";
    public final static String CHANNELS = NAME_SPACE + ".channels";
    public final static String CROP_IMAGE = NAME_SPACE + ".cropimage";
    public final static String IMAGE_LOADER = NAME_SPACE + ".imageloader";

    public BaseImageRecordReader() {}

    public BaseImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator) {
        this(height, width, channels, labelGenerator, null);
    }

    public BaseImageRecordReader(int height, int width, int channels, PathMultiLabelGenerator labelGenerator) {
        this(height, width, channels, null, labelGenerator,null);
    }

    public BaseImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator,
                                 ImageTransform imageTransform) {
        this(height, width, channels, labelGenerator, null, imageTransform);
    }

    protected BaseImageRecordReader(int height, int width, int channels, PathLabelGenerator labelGenerator,
                                    PathMultiLabelGenerator labelMultiGenerator, ImageTransform imageTransform) {
        this.height = height;
        this.width = width;
        this.channels = channels;
        this.labelGenerator = labelGenerator;
        this.labelMultiGenerator = labelMultiGenerator;
        this.imageTransform = imageTransform;
        this.appendLabel = (labelGenerator != null || labelMultiGenerator != null);
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
            imageLoader = new NativeImageLoader(height, width, channels, imageTransform);
        }

        if(split instanceof InputStreamInputSplit) {
            this.inputSplit = split;
            this.finishedInputStreamSplit = false;
            return;
        }

        inputSplit = split;



        URI[] locations = split.locations();
        if (locations != null && locations.length >= 1) {
            if (appendLabel && labelGenerator != null && labelGenerator.inferLabelClasses()) {
                Set<String> labelsSet = new HashSet<>();
                for (URI location : locations) {
                    File imgFile = new File(location);
                    File parentDir = imgFile.getParentFile();
                    String name = labelGenerator.getLabelForPath(location).toString();
                    labelsSet.add(name);
                    if (pattern != null) {
                        String label = name.split(pattern)[patternPosition];
                        fileNameMap.put(imgFile.toString(), label);
                    }
                }
                labels.clear();
                labels.addAll(labelsSet);
            }
            iter = new FileFromPathIterator(inputSplit.locationsPathIterator()); //This handles randomization internally if necessary
        } else
            throw new IllegalArgumentException("No path locations found in the split.");

        if (split instanceof FileSplit) {
            //remove the root directory
            FileSplit split1 = (FileSplit) split;
            labels.remove(split1.getRootDir());
        }

        //To ensure consistent order for label assignment (irrespective of file iteration order), we want to sort the list of labels
        Collections.sort(labels);
    }


    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        this.appendLabel = conf.getBoolean(APPEND_LABEL, appendLabel);
        this.labels = new ArrayList<>(conf.getStringCollection(LABELS));
        this.height = conf.getInt(HEIGHT, height);
        this.width = conf.getInt(WIDTH, width);
        this.channels = conf.getInt(CHANNELS, channels);
        this.cropImage = conf.getBoolean(CROP_IMAGE, cropImage);
        if ("imageio".equals(conf.get(IMAGE_LOADER))) {
            this.imageLoader = new ImageLoader(height, width, channels, cropImage);
        } else {
            this.imageLoader = new NativeImageLoader(height, width, channels, imageTransform);
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
    public void initialize(Configuration conf, InputSplit split, ImageTransform imageTransform)
            throws IOException, InterruptedException {
        this.imageLoader = null;
        this.imageTransform = imageTransform;
        initialize(conf, split);
    }


    @Override
    public List<Writable> next() {
        if(inputSplit instanceof InputStreamInputSplit) {
            InputStreamInputSplit inputStreamInputSplit = (InputStreamInputSplit) inputSplit;
            try {
                NDArrayWritable ndArrayWritable =  new NDArrayWritable(imageLoader.asMatrix(inputStreamInputSplit.getIs()));
                finishedInputStreamSplit = true;
                return Arrays.<Writable>asList(ndArrayWritable);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        if (iter != null) {
            List<Writable> ret;
            File image = iter.next();
            currentFile = image;

            if (image.isDirectory())
                return next();
            try {
                invokeListeners(image);
                INDArray row = imageLoader.asMatrix(image);
                Nd4j.getAffinityManager().ensureLocation(row, AffinityManager.Location.DEVICE);
                ret = RecordConverter.toRecord(row);
                if (appendLabel || writeLabel){
                    if(labelMultiGenerator != null){
                        ret.addAll(labelMultiGenerator.getLabels(image.getPath()));
                    } else {
                        if (labelGenerator.inferLabelClasses()) {
                            //Standard classification use case (i.e., handle String -> integer conversion
                            ret.add(new IntWritable(labels.indexOf(getLabel(image.getPath()))));
                        } else {
                            //Regression use cases, and PathLabelGenerator instances that already map to integers
                            ret.add(labelGenerator.getLabelForPath(image.getPath()));
                        }
                    }
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
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
        if(inputSplit instanceof InputStreamInputSplit) {
            return finishedInputStreamSplit;
        }

        if (iter != null) {
            return iter.hasNext();
        } else if (record != null) {
            return !hitImage;
        }
        throw new IllegalStateException("Indeterminant state: record must not be null, or a file iterator must exist");
    }

    @Override
    public boolean batchesSupported() {
        return (imageLoader instanceof NativeImageLoader);
    }

    @Override
    public List<List<Writable>> next(int num) {
        Preconditions.checkArgument(num > 0, "Number of examples must be > 0: got " + num);

        if (imageLoader == null) {
            imageLoader = new NativeImageLoader(height, width, channels, imageTransform);
        }

        List<File> currBatch = new ArrayList<>();

        int cnt = 0;

        int numCategories = (appendLabel || writeLabel) ? labels.size() : 0;
        List<Integer> currLabels = null;
        List<Writable> currLabelsWritable = null;
        List<List<Writable>> multiGenLabels = null;
        while (cnt < num && iter.hasNext()) {
            currentFile = iter.next();
            currBatch.add(currentFile);
            if (appendLabel || writeLabel) {
                //Collect the label Writables from the label generators
                if(labelMultiGenerator != null){
                    if(multiGenLabels == null)
                        multiGenLabels = new ArrayList<>();

                    multiGenLabels.add(labelMultiGenerator.getLabels(currentFile.getPath()));
                } else {
                    if (labelGenerator.inferLabelClasses()) {
                        if (currLabels == null)
                            currLabels = new ArrayList<>();
                        currLabels.add(labels.indexOf(getLabel(currentFile.getPath())));
                    } else {
                        if (currLabelsWritable == null)
                            currLabelsWritable = new ArrayList<>();
                        currLabelsWritable.add(labelGenerator.getLabelForPath(currentFile.getPath()));
                    }
                }
            }
            cnt++;
        }

        INDArray features = Nd4j.createUninitialized(new int[] {cnt, channels, height, width}, 'c');
        Nd4j.getAffinityManager().tagLocation(features, AffinityManager.Location.HOST);
        for (int i = 0; i < cnt; i++) {
            try {
                ((NativeImageLoader) imageLoader).asMatrixView(currBatch.get(i),
                        features.tensorAlongDimension(i, 1, 2, 3));
            } catch (Exception e) {
                System.out.println("Image file failed during load: " + currBatch.get(i).getAbsolutePath());
                throw new RuntimeException(e);
            }
        }
        Nd4j.getAffinityManager().ensureLocation(features, AffinityManager.Location.DEVICE);


        List<INDArray> ret = new ArrayList<>();
        ret.add(features);
        if (appendLabel || writeLabel) {
            //And convert the previously collected label Writables from the label generators
            if(labelMultiGenerator != null){
                List<Writable> temp = new ArrayList<>();
                List<Writable> first = multiGenLabels.get(0);
                for(int col=0; col<first.size(); col++ ){
                    temp.clear();
                    for (List<Writable> multiGenLabel : multiGenLabels) {
                        temp.add(multiGenLabel.get(col));
                    }
                    INDArray currCol = RecordConverter.toMinibatchArray(temp);
                    ret.add(currCol);
                }
            } else {
                INDArray labels;
                if (labelGenerator.inferLabelClasses()) {
                    //Standard classification use case (i.e., handle String -> integer conversion)
                    labels = Nd4j.create(cnt, numCategories, 'c');
                    Nd4j.getAffinityManager().tagLocation(labels, AffinityManager.Location.HOST);
                    for (int i = 0; i < currLabels.size(); i++) {
                        labels.putScalar(i, currLabels.get(i), 1.0f);
                    }
                } else {
                    //Regression use cases, and PathLabelGenerator instances that already map to integers
                    if (currLabelsWritable.get(0) instanceof NDArrayWritable) {
                        List<INDArray> arr = new ArrayList<>();
                        for (Writable w : currLabelsWritable) {
                            arr.add(((NDArrayWritable) w).get());
                        }
                        labels = Nd4j.concat(0, arr.toArray(new INDArray[arr.size()]));
                    } else {
                        labels = RecordConverter.toMinibatchArray(currLabelsWritable);
                    }
                }

                ret.add(labels);
            }
        }

        return new NDArrayRecordBatch(ret);
    }

    @Override
    public void close() throws IOException {
        //No op
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
        if (fileNameMap != null && fileNameMap.containsKey(path))
            return fileNameMap.get(path);
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
        this.writeLabel = true;
    }

    @Override
    public void reset() {
        if (inputSplit == null)
            throw new UnsupportedOperationException("Cannot reset without first initializing");
        inputSplit.reset();
        if (iter != null) {
            iter = new FileFromPathIterator(inputSplit.locationsPathIterator());
        } else if (record != null) {
            hitImage = false;
        }
    }

    @Override
    public boolean resetSupported(){
        if(inputSplit == null){
            return false;
        }
        return inputSplit.resetSupported();
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
        if (appendLabel)
            ret.add(new IntWritable(labels.indexOf(getLabel(uri.getPath()))));
        return ret;
    }

    @Override
    public Record nextRecord() {
        List<Writable> list = next();
        URI uri = URIUtil.fileToURI(currentFile);
        return new org.datavec.api.records.impl.Record(list, new RecordMetaDataURI(uri, BaseImageRecordReader.class));
    }

    @Override
    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return loadFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    @Override
    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<Record> out = new ArrayList<>();
        for (RecordMetaData meta : recordMetaDatas) {
            URI uri = meta.getURI();
            File f = new File(uri);

            List<Writable> next;
            try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(f)))) {
                next = record(uri, dis);
            }
            out.add(new org.datavec.api.records.impl.Record(next, meta));
        }
        return out;
    }
}
