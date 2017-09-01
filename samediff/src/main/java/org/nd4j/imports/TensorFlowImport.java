package org.nd4j.imports;

import com.google.protobuf.TextFormat;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.tensorflow.framework.GraphDef;

import java.io.*;

/**
 * This class provides TensorFlow graphs & models import
 *
 * @author raver119@gmail.com
 */
public class TensorFlowImport {

    /**
     *
     * @param graphFile
     * @return
     */
    public static SameDiff importGraph(File graphFile) {
        try (FileInputStream fis = new FileInputStream(graphFile); BufferedInputStream bis = new BufferedInputStream(fis)) {
            GraphDef def = GraphDef.parseFrom(bis);
            return importGraph(def);
        } catch (Exception e) {
            try (FileInputStream fis = new FileInputStream(graphFile); BufferedInputStream bis = new BufferedInputStream(fis); BufferedReader reader = new BufferedReader(new InputStreamReader(bis))) {
                GraphDef.Builder builder = GraphDef.newBuilder();

                StringBuilder str = new StringBuilder();
                String line = null;
                while ((line = reader.readLine()) != null) {
                    str.append(line).append("\n");
                }

                TextFormat.getParser().merge(str.toString(), builder);
                GraphDef def = builder.build();
                return importGraph(def);
            } catch (Exception e2) {
                throw new ND4JIllegalStateException("Can't parse graph: unknown format");
            }
        }
    }


    public static SameDiff importGraph(GraphDef tfGraph) {

        return null;
    }
}
