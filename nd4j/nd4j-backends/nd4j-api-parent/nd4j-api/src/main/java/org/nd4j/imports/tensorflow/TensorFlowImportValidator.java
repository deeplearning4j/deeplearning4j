package org.nd4j.imports.tensorflow;

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
 * A simple utility that analyzes TensorFlow graphs and reports details about the models:
 * - The path of the model file(s)
 * - The path of the model(s) that can't be imported due to missing ops
 * - The path of model files that couldn't be read for some reason (corrupt file?)
 * - The total number of ops in all graphs
 * - The number of unique ops in all graphs
 * - The (unique) names of all ops encountered in all graphs
 * - The (unique) names of all ops that were encountered, and can be imported, in all graphs
 * - The (unique) names of all ops that were encountered, and can NOT be imported (lacking import mapping)
 *
 * Note that an op is considered to be importable if has an import mapping specified for that op name in SameDiff.
 * This alone does not guarantee that the op can be imported successfully.
 *
 * @author Alex Black
 */
@Slf4j
public class TensorFlowImportValidator {

    /**
     * Recursively scan the specified directory for .pb files, and evaluate
     * @param directory
     * @return
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

    public static TFImportStatus checkModelForImport(File file) throws IOException {
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
            log.warn("Failed to import model: " + file.getPath(), t);
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
