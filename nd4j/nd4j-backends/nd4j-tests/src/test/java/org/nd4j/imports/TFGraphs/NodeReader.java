package org.nd4j.imports.TFGraphs;

import lombok.NonNull;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.nio.file.Files;

public class NodeReader {
    public static INDArray readArray(@NonNull String graph, @NonNull String variable) throws Exception {
        File shapeFile = null;
        try {
            shapeFile = new ClassPathResource("tf_graphs/examples/" + graph + "/" + variable + ".prediction_inbw.shape").getFile();
        } catch (Exception e) {
            try {
                shapeFile = new ClassPathResource("tf_graphs/examples/" + graph + "/" + variable + ".shape").getFile();
            } catch (Exception e1) {
                throw new RuntimeException(e);
            }
        }

        File valuesFile = null;
        try {
            valuesFile = new ClassPathResource("tf_graphs/examples/" + graph + "/" + variable + ".prediction_inbw.csv").getFile();
        } catch (Exception e) {
            try {
                valuesFile = new ClassPathResource("tf_graphs/examples/" + graph + "/" + variable +".csv").getFile();
            } catch (Exception e1) {
                throw new RuntimeException(e);
            }
        }

        val shapeLines = Files.readAllLines(shapeFile.toPath());
        val valuesLines = Files.readAllLines(valuesFile.toPath());

        val shape = new long[shapeLines.size()];
        val values = new double[valuesLines.size()];
        int cnt = 0;
        for (val v: shapeLines)
            shape[cnt++] = Long.valueOf(v);

        cnt = 0;
        for (val v: valuesLines)
            values[cnt++] = Double.valueOf(v);

        return Nd4j.create(values, shape);
    }
}
