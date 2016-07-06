package org.canova.image.recordreader;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.canova.api.berkeley.Pair;
import org.canova.api.io.data.DoubleWritable;
import org.canova.api.io.data.IntWritable;
import org.canova.api.io.data.Text;
import org.canova.api.io.labels.PathLabelGenerator;
import org.canova.api.split.FileSplit;
import org.canova.api.split.InputSplit;
import org.canova.api.writable.Writable;
import org.canova.common.RecordConverter;
import org.canova.image.loader.BaseImageLoader;
import org.canova.image.loader.ImageLoader;
import org.canova.image.loader.NativeImageLoader;
import org.canova.image.transform.ImageTransform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URI;
import java.util.*;

/**
 * Record reader to handle ImageNet dataset
 **
 * Built to avoid changing api at this time. Api should change to track labels that are only referenced by id in filename
 * Creates a hashmap for label name to id and references that with filename to generate matching lables.
 */
@Deprecated
public class ImageNetRecordReader extends BaseImageRecordReader {

    protected static Logger log = LoggerFactory.getLogger(ImageNetRecordReader.class);
    protected Map<String,String> labelFileIdMap = new LinkedHashMap<>();
    protected String labelPath;
    protected String fileNameMapPath = null; // use when the WNID is not in the filename (e.g. val labels)
    protected boolean eval = false; // use to load label ids for validation data set

    public ImageNetRecordReader(int[] imgDim, PathLabelGenerator labelGenerator, ImageTransform imgTransform, double normalizeValue) {
        this.height = imgDim[0];
        this.width = imgDim[1];
        this.channels = imgDim[2];
        this.eval = true;
    }

    private Map<String, String> defineLabels(String path) throws IOException {
        Map<String,String> tmpMap = new LinkedHashMap<>();
        BufferedReader br = new BufferedReader(new FileReader(path));
        String line;

        while ((line = br.readLine()) != null) {
            String row[] = line.split(",");
            tmpMap.put(row[0], row[1]);
            labels.add(row[1]);
        }
        return tmpMap;
    }

    private void imgNetLabelSetup() throws IOException {
        // creates hashmap with WNID (synset id) as key and first descriptive word in list as the string name
        if (labelPath != null && labelFileIdMap.isEmpty()) {
            labelFileIdMap = defineLabels(labelPath);
            labels = new ArrayList<>(labelFileIdMap.values());
        }
        // creates hasmap with filename as key and WNID(synset id) as value
        if (fileNameMapPath != null && fileNameMap.isEmpty()) {
            fileNameMap = defineLabels(fileNameMapPath);
        }
    }

    @Override
    public void initialize(InputSplit split) throws IOException {
        if (imageLoader == null) {
            imageLoader = new NativeImageLoader(height, width, channels, cropImage);
        }
        inputSplit = split;
        imgNetLabelSetup();

        if(split instanceof FileSplit) {
            URI[] locations = split.locations();
            if(locations != null && locations.length >= 1) {
                if(locations.length > 1) {
                    List<File> allFiles = new ArrayList<>();
                    for(URI location : locations) {
                        File iter = new File(location);
                        if(!iter.isDirectory() && containsFormat(iter.getAbsolutePath()))
                            allFiles.add(iter);
                    }
                    iter =  allFiles.listIterator();
                }
                else {
                    File curr = new File(locations[0]);
                    if(!curr.exists())
                        throw new IllegalArgumentException("Path " + curr.getAbsolutePath() + " does not exist!");
                    if(curr.isDirectory())
                        iter = FileUtils.iterateFiles(curr, null, true);
                    else
                        iter =  Collections.singletonList(curr).listIterator();
                }
            }
        } else {
            throw new UnsupportedClassVersionError("Split needs to be an instance of FileSplit for this record reader.");
        }

    }

    @Override
    public Collection<Writable> next() {
        if(iter != null) {
            File image = iter.next();

            if(image.isDirectory())
                return next();

            try {
                invokeListeners(image);
                return load(imageLoader.asRowVector(image), image.getName());
            } catch (Exception e) {
                e.printStackTrace();
            }

            Collection<Writable> ret = new ArrayList<>();
            if(iter.hasNext()) {
                return ret;
            }
            else {
                if(iter.hasNext()) {
                    try {
                        image = iter.next();
                        invokeListeners(image);
                        ret.add(new Text(FileUtils.readFileToString(image)));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            return ret;
        }
        else if(record != null) {
            hitImage = true;
            invokeListeners(record);
            return record;
        }
        throw new IllegalStateException("No more elements");
    }

    private Collection<Writable> load(INDArray image, String filename) throws IOException {
        int labelId = -1;
        Collection<Writable> ret = RecordConverter.toRecord(image);
        if(appendLabel && fileNameMapPath == null) {
            String WNID = FilenameUtils.getBaseName(filename).split(pattern)[patternPosition];
            labelId = labels.indexOf(labelFileIdMap.get(WNID));
        } else if (eval) {
            String fileName = FilenameUtils.getName(filename); // currently expects file extension
            labelId = labels.indexOf(labelFileIdMap.get(fileNameMap.get(fileName)));
        }
        if (labelId >= 0)
            ret.add(new IntWritable(labelId));
        else
            throw new IllegalStateException("Illegal label " + labelId);
        return ret;
    }

    @Override
    public Collection<Writable> record(URI uri, DataInputStream dataInputStream ) throws IOException {
        invokeListeners(uri);
        imgNetLabelSetup();
        return load(imageLoader.asRowVector(dataInputStream), FilenameUtils.getName(uri.getPath()));
    }
}
