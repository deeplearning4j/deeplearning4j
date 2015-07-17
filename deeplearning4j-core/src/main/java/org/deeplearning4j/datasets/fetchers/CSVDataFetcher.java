/*
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

package org.deeplearning4j.datasets.fetchers;

import au.com.bytecode.opencsv.CSV;
import au.com.bytecode.opencsv.CSVReadProc;
import org.deeplearning4j.berkeley.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * CSV record based data fetcher
 *
 * @author Adam Gibson
 */
@Deprecated
public class CSVDataFetcher extends BaseDataFetcher {

    private  CSV csv;    // new instance is immutable


    private InputStream is;
    private int labelColumn;
    private DataSet all;


    /**
     * Constructs a csv data fetcher with the specified label column
     * skipping no lines
     * @param is the input stream to read from
     * @param labelColumn the index of the column (0 based)
     */
    public CSVDataFetcher(InputStream is,int labelColumn) {
        this.is = is;
        this.labelColumn = labelColumn;
        csv = CSV.skipLines(0)
                .separator(',')  // delimiter of fields
                .quote('"')      // quote character
                .create();       // new instance is immutable

        init();
    }


    /**
     * Constructs a csv data fetcher with the specified
     * label column skipping no lines
     * @param f the file to read from
     * @param labelColumn the index of the column (0 based)
     * @throws IOException
     */
    public CSVDataFetcher(File f,int labelColumn) throws IOException {
        this(new BufferedInputStream(new FileInputStream(f)),labelColumn,0);
    }

    /**
     * Constructs a csv data fetcher with the specified number of lines to skip
     * @param is the input stream to read from
     * @param labelColumn the index of the column (0 based)
     * @param skipLines the number of lines to skip
     */
    public CSVDataFetcher(InputStream is,int labelColumn,int skipLines) {
        this.is = is;
        this.labelColumn = labelColumn;
        csv = CSV.skipLines(skipLines)
                .separator(',')  // delimiter of fields
                .noQuote()     // quote character
                .create();       // new instance is immutable

        init();
    }

    /**
     * Constructs a csv data fetcher with the specified number of lines to skip
     * @param f the file to read from
     * @param labelColumn the index of the column (0 based)
     * @param skipLines the number of lines to skip
     * @throws IOException
     */
    public CSVDataFetcher(File f,int labelColumn,int skipLines) throws IOException {
        this(new BufferedInputStream(new FileInputStream(f)),labelColumn,skipLines);
    }


    private void init() {
        final Set<String> labels = new HashSet<>();
        final List<String> rowLabels = new ArrayList<>();
        final List<INDArray> features = new ArrayList<>();
        final AtomicInteger i1 = new AtomicInteger(-1);
        csv.read(is,new CSVReadProc() {
            @Override
            public void procRow(int rowIndex, String... values) {
                if(values.length < 1)
                    return;
                if(i1.get() < 1) {
                    i1.set(values.length - 1);
                    CSVDataFetcher.this.inputColumns = values.length - 1;

                }
                else if(values.length  - 1 != i1.get())
                    return;
                Pair<INDArray,String> row = processRow(values);
                rowLabels.add(row.getSecond());
                labels.add(row.getSecond());
                features.add(row.getFirst());
            }
        });

        List<DataSet> l = new ArrayList<>();
        List<String> labelIndices = new ArrayList<>(labels);
        for(int i = 0; i < rowLabels.size(); i++) {
            l.add(new DataSet(features.get(i), FeatureUtil.toOutcomeVector(labelIndices.indexOf(rowLabels.get(i)), labels.size())));
        }

        this.numOutcomes = labels.size();
        this.totalExamples = l.size();
        
        all = DataSet.merge(l);
    }


    private Pair<INDArray,String> processRow(String[] data) {

        String label = data[labelColumn].replaceAll(".\".","");

        double[] d = new double[data.length - 1];
        int index = 0;
        for(int i = 0; i < data.length; i++){
            if(i != labelColumn){
                d[index] = Double.parseDouble(data[i]);
                index ++;
            }
        }
        
        INDArray d1 = Nd4j.create(d).reshape(1,d.length);
        return new Pair<>(d1, label);
    }

    /**
     * Fetches the next dataset. You need to call this
     * to getFromOrigin a new dataset, otherwise {@link #next()}
     * just returns the last data applyTransformToDestination fetch
     *
     * @param numExamples the number of examples to fetch
     */
    @Override
    public void fetch(int numExamples) {
        int end = cursor + numExamples;
        if(end >= all.numExamples())
            end = all.numExamples();
        initializeCurrFromList(all.asList().subList(cursor,end));
        cursor += numExamples;
    }
}
