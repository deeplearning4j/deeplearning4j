package org.nd4j.imports.tensorflow;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.io.*;
import java.util.*;

/**
 * A simple utility that analyzes TensorFlow graphs and reports details about the models:<br>
 * - The path of the model file(s)<br>
 * - The path of the model(s) that can't be imported due to missing ops<br>
 * - The path of model files that couldn't be read for some reason (corrupt file?)<br>
 * - The total number of ops in all graphs<br>
 * - The number of unique ops in all graphs<br>
 * - The (unique) names of all ops encountered in all graphs<br>
 * - The (unique) names of all ops that were encountered, and can be imported, in all graphs<br>
 * - The (unique) names of all ops that were encountered, and can NOT be imported (lacking import mapping)<br>
 *<br>
 * Note that an op is considered to be importable if has an import mapping specified for that op name in SameDiff.<br>
 * This alone does not guarantee that the op can be imported successfully.
 *
 * @author Alex Black
 */
@Slf4j
public class TensorFlowImportValidator {

    /**
     * Recursively scan the specified directory for .pb files, and evaluate which operations/graphs can/can't be imported
     * @param directory Directory to scan
     * @return Status for TensorFlow import for all models in
     * @throws IOException
     */
    public static TFImportStatus checkAllModelsForImport(File directory) throws IOException {
        Preconditions.checkState(directory.isDirectory(), "Specified directory %s is not actually a directory", directory);

        Collection<File> files = FileUtils.listFiles(directory, new String[]{"pb"}, true);
        Preconditions.checkState(!files.isEmpty(), "No .pb files found in directory %s", directory);

        TFImportStatus status = null;
        for(File f : files){
            if(status == null){
                status = checkModelForImport(f);
            } else {
                status = status.merge(checkModelForImport(f));
            }
        }
        return status;
    }

    /**
     * See {@link #checkModelForImport(File)}. Defaults to exceptionOnRead = false
     */
    public static TFImportStatus checkModelForImport(@NonNull File file) throws IOException {
        return checkModelForImport(file, false);
    }

    /**
     * Check whether the TensorFlow frozen model (protobuf format) can be imported into SameDiff or not
     * @param file            Protobuf file
     * @param exceptionOnRead If true, and the file can't be read, throw an exception. If false, return an "empty" TFImportStatus
     * @return Status for importing the file
     * @throws IOException If error
     */
    public static TFImportStatus checkModelForImport(@NonNull File file, boolean exceptionOnRead) throws IOException {
        TFGraphMapper m = TFGraphMapper.getInstance();

        try {
            int opCount = 0;
            Set<String> opNames = new HashSet<>();
            try (InputStream is = new BufferedInputStream(new FileInputStream(file))) {
                GraphDef graphDef = m.parseGraphFrom(is);
                List<NodeDef> nodes = m.getNodeList(graphDef);
                for (NodeDef nd : nodes) {
                    if(m.isVariableNode(nd) || m.isPlaceHolderNode(nd))
                        continue;

                    String op = nd.getOp();
//                System.out.println(op);
                    opNames.add(op);
                    opCount++;
                }
            }

            Set<String> importSupportedOpNames = new HashSet<>();
            Set<String> unsupportedOpNames = new HashSet<>();

            for (String s : opNames) {
                if (DifferentialFunctionClassHolder.getInstance().getOpWithTensorflowName(s) != null) {
                    importSupportedOpNames.add(s);
                } else {
                    unsupportedOpNames.add(s);
                }
            }

            return new TFImportStatus(
                    Collections.singletonList(file.getPath()),
                    unsupportedOpNames.size() > 0 ? Collections.singletonList(file.getPath()) : Collections.<String>emptyList(),
                    Collections.<String>emptyList(),
                    opCount,
                    opNames.size(),
                    opNames,
                    importSupportedOpNames,
                    unsupportedOpNames);
        } catch (Throwable t){
            if(exceptionOnRead) {
                throw new IOException("Error reading model from path " + file.getPath() + " - not a TensorFlow frozen model in ProtoBuf format?", t);
            }
            log.warn("Failed to import model from file: " + file.getPath() + " - not a TensorFlow frozen model in ProtoBuf format?", t);
            return new TFImportStatus(
                    Collections.<String>emptyList(),
                    Collections.<String>emptyList(),
                    Collections.singletonList(file.getPath()),
                    0,
                    0,
                    Collections.<String>emptySet(),
                    Collections.<String>emptySet(),
                    Collections.<String>emptySet());
        }
    }
}
