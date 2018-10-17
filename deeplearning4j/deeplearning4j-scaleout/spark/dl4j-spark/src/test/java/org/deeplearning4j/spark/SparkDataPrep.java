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

package org.deeplearning4j.spark;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.listener.RecordListener;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.FileBatchRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.api.loader.DataSetLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.api.loader.FileBatch;
import org.nd4j.api.loader.Source;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.dataset.DataSet;

import java.io.*;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

public class SparkDataPrep {

    public static void batchAndExportFiles(File inputDirectory, boolean recursive, File outputDirectory, int batchSize) throws IOException {
        batchAndExportFiles(inputDirectory, null, recursive, outputDirectory, batchSize);
    }

    public static void batchAndExportFiles(File inputDirectory, String[] extensions, boolean recursive, File outputDirectory, int batchSize) throws IOException {
        //Local version
        List<File> c = new ArrayList<>(FileUtils.listFiles(inputDirectory, extensions, true));
        Collections.shuffle(c);

        //Construct file batch
        List<String> list = new ArrayList<>();
        List<byte[]> bytes = new ArrayList<>();
        for( int i=0; i<c.size(); i++ ){
            list.add(c.get(i).getPath());
            bytes.add(FileUtils.readFileToByteArray(c.get(i)));

            if(list.size() == batchSize){
                process(list, bytes, outputDirectory);
            }
        }
        if(list.size() > 0){
            process(list, bytes, outputDirectory);
        }
    }

    private static void process(List<String> paths, List<byte[]> bytes, File outputDirectory) throws IOException {
        FileBatch fb = new FileBatch(bytes, paths);
        String name = UUID.randomUUID().toString().replaceAll("-","") + ".zip";
        File f = new File(outputDirectory, name);
        try (BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f))) {
            fb.writeAsZip(bos);
        }

        paths.clear();
        bytes.clear();
    }

    public void batchAndExportFiles(JavaRDD<String> filePaths, final String rootOutputDir, final int batchSize, JavaSparkContext sc){

        final org.apache.hadoop.conf.Configuration hadoopConfig = sc.hadoopConfiguration();
        //Here: assume input is images. We can't store them as Float32 arrays - that's too inefficient
        // instead: let's store the raw file content in a batch.
        long count = filePaths.count();
        long maxPartitions = count / batchSize;
        JavaRDD<String> repartitioned = filePaths.repartition(Math.max(filePaths.getNumPartitions(), (int)maxPartitions));
        repartitioned.foreachPartition(new VoidFunction<Iterator<String>>() {
            @Override
            public void call(Iterator<String> stringIterator) throws Exception {
                //Construct file batch
                List<String> list = new ArrayList<>();
                List<byte[]> bytes = new ArrayList<>();
                while(stringIterator.hasNext()){

                    if(list.size() == batchSize){
                        process(list, bytes);
                    }
                }
                if(list.size() > 0){
                    process(list, bytes);
                }
            }

            private void process(List<String> paths, List<byte[]> bytes) throws IOException {
                FileBatch fb = new FileBatch(bytes, paths);
                String name = UUID.randomUUID().toString().replaceAll("-","") + ".zip";
                String outPath = FilenameUtils.concat(rootOutputDir, name);
                FileSystem fileSystem = FileSystem.get(hadoopConfig);
                try (BufferedOutputStream bos = new BufferedOutputStream(fileSystem.create(new Path(outPath)))) {
                    fb.writeAsZip(bos);
                }

                paths.clear();
                bytes.clear();
            }
        });
    }



//    @AllArgsConstructor
//    @Data
//    public static class FileBatch implements Serializable {
//        public static final String ORIG_PATHS_FILENAME = "originalPaths.txt";
//
//        private final List<byte[]> fileBytes;
//        private final List<String> originalPaths;
//
//        public void writeAsZip(OutputStream os) throws IOException {
//            try(ZipOutputStream zos = new ZipOutputStream(new BufferedOutputStream(os))){
//
//                //Write original paths as a text file:
//                ZipEntry ze = new ZipEntry(ORIG_PATHS_FILENAME);
//                String originalPathsJoined = StringUtils.join(originalPaths, "\n"); //Java String.join is Java 8
//                zos.putNextEntry(ze);
//                zos.write(originalPathsJoined.getBytes(StandardCharsets.UTF_8));
//
//                for( int i=0; i<fileBytes.size(); i++ ){
//                    String name = "file_" + i + ".bin";
//                    ze = new ZipEntry(name);
//                    zos.putNextEntry(ze);
//                    zos.write(fileBytes.get(i));
//                }
//            }
//        }
//
//        public static FileBatch readFromZip(InputStream is) throws IOException {
//            String originalPaths = null;
//            Map<Integer,byte[]> bytesMap = new HashMap<>();
//            try(ZipInputStream zis = new ZipInputStream(new BufferedInputStream(is))){
//                ZipEntry ze;
//                while((ze = zis.getNextEntry()) != null){
//                    String name = ze.getName();
//                    long size = ze.getSize();
//                    byte[] bytes = new byte[(int)size];
//                    zis.read(bytes);
//                    if(name.equals(ORIG_PATHS_FILENAME)){
//                        originalPaths = new String(bytes, 0, bytes.length, StandardCharsets.UTF_8);
//                    } else {
//                        int idxSplit = name.indexOf("_");
//                        int idxSplit2 = name.indexOf(".");
//                        int fileIdx = Integer.parseInt(name.substring(idxSplit+1, idxSplit2));
//                        bytesMap.put(fileIdx, bytes);
//                    }
//                }
//            }
//
//            List<byte[]> list = new ArrayList<>(bytesMap.size());
//            for(int i=0; i<bytesMap.size(); i++ ){
//                list.add(bytesMap.get(i));
//            }
//
//            List<String> origPaths = Arrays.asList(originalPaths.split("\n"));
//            return new FileBatch(list, origPaths);
//        }
//    }

    public static class RecordReaderFileBatchDataSetLoader implements DataSetLoader {
        private final RecordReader recordReader;
        private final int batchSize;
        private final int labelIndexFrom;
        private final int labelIndexTo;
        private final int numPossibleLabels;
        private final boolean regression;

        public RecordReaderFileBatchDataSetLoader(RecordReader recordReader, int batchSize, int labelIndex, int numClasses){
            this(recordReader, batchSize, labelIndex, labelIndex, numClasses, false);
        }

        public RecordReaderFileBatchDataSetLoader(RecordReader recordReader, int batchSize, int labelIndexFrom, int labelIndexTo,
                                           int numPossibleLabels, boolean regression) {
            this.recordReader = recordReader;
            this.batchSize = batchSize;
            this.labelIndexFrom = labelIndexFrom;
            this.labelIndexTo = labelIndexTo;
            this.numPossibleLabels = numPossibleLabels;
            this.regression = regression;
        }

        @Override
        public DataSet load(Source source) throws IOException {
            FileBatch fb = FileBatch.readFromZip(source.getInputStream());

            //Wrap file batch in RecordReader
            //Create RecordReaderDataSetIterator
            //Return dataset
            RecordReader rr = new FileBatchRecordReader(recordReader, fb);
            RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(rr, null, batchSize, labelIndexFrom, labelIndexTo, numPossibleLabels, -1, regression);
            DataSet ds = iter.next();
            return ds;
        }
    }

//    public static class FileBatchRecordReader implements RecordReader {
//
//        private final RecordReader recordReader;
//        private final FileBatch fileBatch;
//        private int position = 0;
//
//        public FileBatchRecordReader(RecordReader rr, FileBatch fileBatch){
//            this.recordReader = rr;
//            this.fileBatch = fileBatch;
//        }
//
//
//        @Override
//        public void initialize(InputSplit split) throws IOException, InterruptedException {
//            //No op
//        }
//
//        @Override
//        public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
//            //No op
//        }
//
//        @Override
//        public boolean batchesSupported() {
//            return false;
//        }
//
//        @Override
//        public List<List<Writable>> next(int num) {
//            List<List<Writable>> out = new ArrayList<>(Math.min(num, 10000));
//            for( int i=0; i<num && hasNext(); i++ ){
//                out.add(next());
//            }
//            return out;
//        }
//
//        @Override
//        public List<Writable> next() {
//            Preconditions.checkState(hasNext(), "No next element");
//
//            byte[] fileBytes = fileBatch.getFileBytes().get(position);
//            String origPath = fileBatch.getOriginalPaths().get(position);
//
//            List<Writable> out;
//            try {
//                out = recordReader.record(URI.create(origPath), new DataInputStream(new ByteArrayInputStream(fileBytes)));
//            } catch (IOException e){
//                throw new RuntimeException("Error reading from file bytes");
//            }
//
//            position++;
//            return out;
//        }
//
//        @Override
//        public boolean hasNext() {
//            return position < fileBatch.getFileBytes().size();
//        }
//
//        @Override
//        public List<String> getLabels() {
//            return recordReader.getLabels();
//        }
//
//        @Override
//        public void reset() {
//            position = 0;
//        }
//
//        @Override
//        public boolean resetSupported() {
//            return true;
//        }
//
//        @Override
//        public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
//            throw new UnsupportedOperationException("Not supported");
//        }
//
//        @Override
//        public Record nextRecord() {
//            return new org.datavec.api.records.impl.Record(next(), null);
//        }
//
//        @Override
//        public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
//            return recordReader.loadFromMetaData(recordMetaData);
//        }
//
//        @Override
//        public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
//            return recordReader.loadFromMetaData(recordMetaDatas);
//        }
//
//        @Override
//        public List<RecordListener> getListeners() {
//            return null;
//        }
//
//        @Override
//        public void setListeners(RecordListener... listeners) {
//            recordReader.setListeners(listeners);
//        }
//
//        @Override
//        public void setListeners(Collection<RecordListener> listeners) {
//            recordReader.setListeners(listeners);
//        }
//
//        @Override
//        public void close() throws IOException {
//            recordReader.close();
//        }
//
//        @Override
//        public void setConf(Configuration conf) {
//            recordReader.setConf(conf);
//        }
//
//        @Override
//        public Configuration getConf() {
//            return recordReader.getConf();
//        }
//    }


}
