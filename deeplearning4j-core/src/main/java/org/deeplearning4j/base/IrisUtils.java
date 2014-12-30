package org.deeplearning4j.base;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.commons.io.IOUtils;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

public class IrisUtils {
    private static final int NUM_FEATURES = 4;
    private static final int NUM_LABELS = 3;

    /**
     * A loader function for the Fischer Iris dataset.
     * @param from beginning row
     * @param to ending row
     * @return a collection of dataset objects between the rows {@code from} and {@code to}
     * @throws IOException
     */
    public static List<DataSet> loadIris(int from, int to) throws IOException {
        assert from < to;
        ClassPathResource resource = new ClassPathResource("/iris.dat");
        List<String> lines = IOUtils.readLines(resource.getInputStream());
        List<DataSet> list = Lists.newArrayList();
        INDArray ret = Nd4j.ones(to - from, NUM_FEATURES);
        Map<String, Integer> outcomeTypes = Maps.newHashMap();
        double[][] outcomes = new double[lines.size()][NUM_LABELS];
        int outcomeIndex = 0;

        for(int i = from; i < to; i++) {
            String line = lines.get(i);
            String[] split = line.split(",");
            addRow(ret, i, split);
            String outcome = split[split.length - 1];
            if(!outcomeTypes.containsKey(outcome)) {
                outcomeTypes.put(outcome, outcomeIndex++);
            }
            double[] rowOutcome = new double[NUM_LABELS];
            rowOutcome[outcomeTypes.get(outcome)] = 1;
            outcomes[i] = rowOutcome;
        }

        for(int i = 0; i < ret.rows(); i++)
            list.add(new DataSet(ret.getRow(i), Nd4j.create(outcomes[i])));

        return list;
    }

    private static void addRow(INDArray ret, int row, String[] line) {
        double[] vector = new double[NUM_FEATURES];
        for(int i = 0; i < NUM_FEATURES; i++)
            vector[i] = Double.parseDouble(line[i]);
        ret.putRow(row,Nd4j.create(vector));
    }
}
