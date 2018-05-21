package org.deeplearning4j.graph.models.loader;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.graph.models.GraphVectors;
import org.deeplearning4j.graph.models.deepwalk.DeepWalk;
import org.deeplearning4j.graph.models.embeddings.GraphVectorsImpl;
import org.deeplearning4j.graph.models.embeddings.InMemoryGraphLookupTable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**GraphVectorSerializer: Provide static methods to save and load DeepWalk/Graph vectors
 *
 */
public class GraphVectorSerializer {
    private static final Logger log = LoggerFactory.getLogger(GraphVectorSerializer.class);
    private static final String DELIM = "\t";

    private GraphVectorSerializer() {}

    public static void writeGraphVectors(DeepWalk deepWalk, String path) throws IOException {

        int nVertices = deepWalk.numVertices();
        int vectorSize = deepWalk.getVectorSize();

        try (BufferedWriter write = new BufferedWriter(new FileWriter(new File(path), false))) {
            for (int i = 0; i < nVertices; i++) {
                StringBuilder sb = new StringBuilder();
                sb.append(i);
                INDArray vec = deepWalk.getVertexVector(i);
                for (int j = 0; j < vectorSize; j++) {
                    double d = vec.getDouble(j);
                    sb.append(DELIM).append(d);
                }
                sb.append("\n");
                write.write(sb.toString());
            }
        }

        log.info("Wrote {} vectors of length {} to: {}", nVertices, vectorSize, path);
    }

    public static GraphVectors loadTxtVectors(File file) throws IOException {

        List<double[]> vectorList = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            LineIterator iter = IOUtils.lineIterator(reader);

            while (iter.hasNext()) {
                String line = iter.next();
                String[] split = line.split(DELIM);
                double[] vec = new double[split.length - 1];
                for (int i = 1; i < split.length; i++) {
                    vec[i - 1] = Double.parseDouble(split[i]);
                }
                vectorList.add(vec);
            }
        }

        int vecSize = vectorList.get(0).length;
        int nVertices = vectorList.size();

        INDArray vectors = Nd4j.create(nVertices, vecSize);
        for (int i = 0; i < vectorList.size(); i++) {
            double[] vec = vectorList.get(i);
            for (int j = 0; j < vec.length; j++) {
                vectors.put(i, j, vec[j]);
            }
        }

        InMemoryGraphLookupTable table = new InMemoryGraphLookupTable(nVertices, vecSize, null, 0.01);
        table.setVertexVectors(vectors);

        return new GraphVectorsImpl<>(null, table);
    }

}
