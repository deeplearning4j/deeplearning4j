package org.nd4j.imports;

import com.google.protobuf.TextFormat;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.GraphDef;

import java.io.*;

/**
 * This class provides TensorFlow graphs & models import
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class TensorFlowImport {

    /**
     *
     * @param graphFile
     * @return
     */
    public static SameDiff importGraph(File graphFile) {
        return new TFGraphMapper().importGraph(graphFile);
    }

    /**
     * This method converts given TF
     * @param tfGraph
     * @return
     */
    public static SameDiff importGraph(GraphDef tfGraph) {
        return new TFGraphMapper().mapGraph(tfGraph);
    }

    /**
     * This method returns intermediate representation from TF protobuf file
     *
     * @param graphFile
     * @return
     */
    public static TGraph importIntermediate(File graphFile) {
        GraphDef def = null;
        try (FileInputStream fis = new FileInputStream(graphFile); BufferedInputStream bis = new BufferedInputStream(fis)) {
            def = GraphDef.parseFrom(bis);
        } catch (Exception e) {
            try (FileInputStream fis2 = new FileInputStream(graphFile); BufferedInputStream bis2 = new BufferedInputStream(fis2); BufferedReader reader = new BufferedReader(new InputStreamReader(bis2))) {
                GraphDef.Builder builder = GraphDef.newBuilder();

                StringBuilder str = new StringBuilder();
                String line = null;
                while ((line = reader.readLine()) != null) {
                    str.append(line);//.append("\n");
                }

                TextFormat.getParser().merge(str.toString(), builder);
                def = builder.build();
            } catch (Exception e2) {
                //
            }
        }

        if (def == null)
            throw new ND4JIllegalStateException("Unknown format");


        return new TFGraphMapper().importIntermediate(def);
    }

}
