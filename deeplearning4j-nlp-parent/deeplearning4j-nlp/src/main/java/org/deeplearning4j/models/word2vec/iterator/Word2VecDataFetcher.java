/*-
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.models.word2vec.iterator;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.movingwindow.Window;
import org.deeplearning4j.text.movingwindow.WindowConverter;
import org.deeplearning4j.text.movingwindow.Windows;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.DataSetFetcher;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;

/**
 *
 */
public class Word2VecDataFetcher implements DataSetFetcher {

    /**
     * 
     */
    private static final long serialVersionUID = 3245955804749769475L;
    private transient Iterator<File> files;
    private Word2Vec vec;
    private static Pattern begin = Pattern.compile("<[A-Z]+>");
    private static Pattern end = Pattern.compile("</[A-Z]+>");
    private List<String> labels = new ArrayList<>();
    private int batch;
    private List<Window> cache = new ArrayList<>();
    private static final Logger log = LoggerFactory.getLogger(Word2VecDataFetcher.class);
    private int totalExamples;
    private String path;

    public Word2VecDataFetcher(String path, Word2Vec vec, List<String> labels) {
        if (vec == null || labels == null || labels.isEmpty())
            throw new IllegalArgumentException(
                            "Unable to initialize due to missing argument or empty label applyTransformToDestination");
        this.vec = vec;
        this.labels = labels;
        this.path = path;
    }



    private DataSet fromCache() {
        INDArray outcomes = null;
        INDArray input = null;
        input = Nd4j.create(batch, vec.lookupTable().layerSize() * vec.getWindow());
        outcomes = Nd4j.create(batch, labels.size());
        for (int i = 0; i < batch; i++) {
            input.putRow(i, WindowConverter.asExampleMatrix(cache.get(i), vec));
            int idx = labels.indexOf(cache.get(i).getLabel());
            if (idx < 0)
                idx = 0;
            outcomes.putRow(i, FeatureUtil.toOutcomeVector(idx, labels.size()));
        }
        return new DataSet(input, outcomes);

    }

    @Override
    public DataSet next() {
        //pop from cache when possible, or when there's nothing left
        if (cache.size() >= batch || !files.hasNext())
            return fromCache();



        File f = files.next();
        try {
            LineIterator lines = FileUtils.lineIterator(f);
            INDArray outcomes = null;
            INDArray input = null;

            while (lines.hasNext()) {
                List<Window> windows = Windows.windows(lines.nextLine());
                if (windows.isEmpty() && lines.hasNext())
                    continue;

                if (windows.size() < batch) {
                    input = Nd4j.create(windows.size(), vec.lookupTable().layerSize() * vec.getWindow());
                    outcomes = Nd4j.create(batch, labels.size());
                    for (int i = 0; i < windows.size(); i++) {
                        input.putRow(i, WindowConverter.asExampleMatrix(cache.get(i), vec));
                        int idx = labels.indexOf(windows.get(i).getLabel());
                        if (idx < 0)
                            idx = 0;
                        INDArray outcomeRow = FeatureUtil.toOutcomeVector(idx, labels.size());
                        outcomes.putRow(i, outcomeRow);
                    }
                    return new DataSet(input, outcomes);


                } else {
                    input = Nd4j.create(batch, vec.lookupTable().layerSize() * vec.getWindow());
                    outcomes = Nd4j.create(batch, labels.size());
                    for (int i = 0; i < batch; i++) {
                        input.putRow(i, WindowConverter.asExampleMatrix(cache.get(i), vec));
                        int idx = labels.indexOf(windows.get(i).getLabel());
                        if (idx < 0)
                            idx = 0;
                        INDArray outcomeRow = FeatureUtil.toOutcomeVector(idx, labels.size());
                        outcomes.putRow(i, outcomeRow);
                    }
                    //add left over to cache; need to ensure that only batch rows are returned
                    /*
                     * Note that I'm aware of possible concerns for sentence sequencing.
                     * This is a hack right now in place of something
                     * that will be way more elegant in the future.
                     */
                    if (windows.size() > batch) {
                        List<Window> leftOvers = windows.subList(batch, windows.size());
                        cache.addAll(leftOvers);
                    }
                    return new DataSet(input, outcomes);
                }

            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return null;
    }



    @Override
    public int totalExamples() {
        return totalExamples;
    }

    @Override
    public int inputColumns() {
        return vec.lookupTable().layerSize() * vec.getWindow();
    }

    @Override
    public int totalOutcomes() {
        return labels.size();

    }

    @Override
    public void reset() {
        files = FileUtils.iterateFiles(new File(path), null, true);
        cache.clear();

    }



    @Override
    public int cursor() {
        return 0;

    }



    @Override
    public boolean hasMore() {
        return files.hasNext() || !cache.isEmpty();
    }

    @Override
    public void fetch(int numExamples) {
        this.batch = numExamples;
    }

    public Iterator<File> getFiles() {
        return files;
    }

    public Word2Vec getVec() {
        return vec;
    }

    public static Pattern getBegin() {
        return begin;
    }

    public static Pattern getEnd() {
        return end;
    }

    public List<String> getLabels() {
        return labels;
    }

    public int getBatch() {
        return batch;
    }

    public List<Window> getCache() {
        return cache;
    }



}
