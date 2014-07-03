package org.deeplearning4j.datasets.fetchers;

import au.com.bytecode.opencsv.CSV;
import au.com.bytecode.opencsv.CSVReadProc;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * CSV record based data fetcher
 *
 * @author Adam Gibson
 */
public class CSVDataFetcher extends BaseDataFetcher {

    private  CSV csv;    // new instance is immutable


    private InputStream is;
    private int labelColumn;
    private DataSet all;

    public CSVDataFetcher(InputStream is,int labelColumn) {
        this.is = is;
        this.labelColumn = labelColumn;
        csv = CSV
                .separator(',')  // delimiter of fields
                .quote('"')      // quote character
                .create();       // new instance is immutable

        init();
    }


    public CSVDataFetcher(File f,int labelColumn) throws IOException {
        this(new BufferedInputStream(new FileInputStream(f)),labelColumn);
    }


    private void init() {
        final Set<Integer> labels = new HashSet<>();
        final List<Integer> rowLabels = new ArrayList<>();
        final List<DoubleMatrix> features = new ArrayList<>();
        csv.read(is,new CSVReadProc() {
            @Override
            public void procRow(int rowIndex, String... values) {
                Pair<DoubleMatrix,Integer> row = processRow(values);
                rowLabels.add(row.getSecond());
                labels.add(row.getSecond());
                features.add(row.getFirst());
            }
        });

        List<DataSet> l = new ArrayList<>();
        for(int i = 0; i < rowLabels.size(); i++) {
            l.add(new DataSet(features.get(i),MatrixUtil.toOutcomeVector(rowLabels.get(i),rowLabels.size())));
        }

        all = DataSet.merge(l);


    }


    private Pair<DoubleMatrix,Integer> processRow(String[] data) {
        int labelVal = Integer.parseInt(data[labelColumn]);
        List<Double> vals = new ArrayList<>();
        for(int i = 0; i < data.length; i++)
            if(i != labelVal)
                vals.add(Double.parseDouble(data[i]));

        double[] d = new double[vals.size()];
        DoubleMatrix d1 = new DoubleMatrix(d).reshape(1,d.length);
        return new Pair<>(d1,labelVal);

    }

    /**
     * Fetches the next dataset. You need to call this
     * to get a new dataset, otherwise {@link #next()}
     * just returns the last data set fetch
     *
     * @param numExamples the number of examples to fetch
     */
    @Override
    public void fetch(int numExamples) {
        int end = cursor + numExamples;
        if(end >= all.numExamples())
            end = all.numExamples();
        initializeCurrFromList(all.asList().subList(cursor,end));
    }
}
