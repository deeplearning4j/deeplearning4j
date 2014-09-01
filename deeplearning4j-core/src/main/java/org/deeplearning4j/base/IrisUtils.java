package org.deeplearning4j.base;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.berkeley.Pair;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.springframework.core.io.ClassPathResource;



public class IrisUtils {


    public static List<DataSet> loadIris(int from,int to) throws IOException {
        ClassPathResource resource = new ClassPathResource("/iris.dat");
        @SuppressWarnings("unchecked")
        List<String> lines = IOUtils.readLines(resource.getInputStream());
        List<DataSet> list = new ArrayList<>();
        INDArray ret = NDArrays.ones(to, 4);
        List<String> outcomeTypes = new ArrayList<>();
        double[][] outcomes = new double[lines.size()][3];
        for(int i = from; i < to; i++) {
            String line = lines.get(i);
            String[] split = line.split(",");

            addRow(ret,i,split);

            String outcome = split[split.length - 1];
            if(!outcomeTypes.contains(outcome))
                outcomeTypes.add(outcome);
            double[] rowOutcome = new double[3];
            rowOutcome[outcomeTypes.indexOf(outcome)] = 1;
            outcomes[i] = rowOutcome;
        }

        for(int i = 0; i < ret.rows(); i++)
            list.add(new DataSet(ret.getRow(i), NDArrays.create(outcomes[i])));



        return list;
    }







    private static void addRow(INDArray ret,int row,String[] line) {
        double[] vector = new double[4];
        for(int i = 0; i < 4; i++)
            vector[i] = Double.parseDouble(line[i]);

        ret.putRow(row,NDArrays.create(vector));
    }
}
