package org.deeplearning4j.datasets.iterator.impl;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.MovingWindowMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * Moving window data fetcher. Handles rotation of matrices in all directions
 * to generate more examples.
 *
 *
 * @author Adam Gibson
 */
public class MovingWindowDataSetFetcher extends BaseDataFetcher  {

    private DataSet data;
    private int windowRows = 28,windowColumns = 28;
    private int cursor = 0;

    public MovingWindowDataSetFetcher(DataSet data,int windowRows,int windowColumns) {
        this.data = data;
        this.windowRows = windowRows;
        this.windowColumns = windowColumns;
        List<DataSet> list = data.asList();
        List<DataSet> flipped = new ArrayList<>();
        for(int i = 0; i < list.size(); i++) {
            DoubleMatrix label = list.get(i).getSecond();
            List<DoubleMatrix> windows = new MovingWindowMatrix(list.get(i).getFirst(),windowRows,windowColumns,true).windows(true);
            for(int j = 0; j < windows.size(); j++) {
                flipped.add(new DataSet(windows.get(i),label));
            }
            flipped.add(list.get(i));
        }

        this.data = DataSet.merge(flipped);

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
          initializeCurrFromList(data.get(MatrixUtil.toIndices(RangeUtils.interval(cursor,cursor + numExamples))).asList());

    }
}
