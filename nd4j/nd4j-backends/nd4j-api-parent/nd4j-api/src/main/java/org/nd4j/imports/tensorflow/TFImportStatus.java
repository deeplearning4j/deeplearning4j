package org.nd4j.imports.tensorflow;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@AllArgsConstructor
@Data
public class TFImportStatus {

    /** The paths of the model(s) that have been investigated */
    private final List<String> modelPaths;
    /** The paths of the models that can't be imported, because they have 1 or more missing ops */
    private final List<String> cantImportModelPaths;
    /** The paths of the models that can't be read for some reason (corruption, etc?) */
    private final List<String> readErrorModelPaths;
    /** The total number of ops in all graphs */
    private final int totalNumOps;
    /** The number of unique ops in all graphs */
    private final int numUniqueOps;
    /** The (unique) names of all ops encountered in all graphs */
    private final Set<String> opNames;
    /** The (unique) names of all ops that were encountered, and can be imported, in all graphs */
    private final Set<String> importSupportedOpNames;
    /** The (unique) names of all ops that were encountered, and can NOT be imported (lacking import mapping) */
    private final Set<String> unsupportedOpNames;


    public TFImportStatus merge(@NonNull TFImportStatus other){
        List<String> newModelPaths = new ArrayList<>(modelPaths);
        newModelPaths.addAll(other.modelPaths);

        List<String> newCantImportModelPaths = new ArrayList<>(cantImportModelPaths);
        newCantImportModelPaths.addAll(other.cantImportModelPaths);

        List<String> newReadErrorModelPaths = new ArrayList<>(readErrorModelPaths);
            newReadErrorModelPaths.addAll(other.readErrorModelPaths);



        Set<String> newOpNames = new HashSet<>(opNames);
        newOpNames.addAll(other.opNames);

        Set<String> newImportSupportedOpNames = new HashSet<>(importSupportedOpNames);
        newImportSupportedOpNames.addAll(other.importSupportedOpNames);

        Set<String> newUnsupportedOpNames = new HashSet<>(unsupportedOpNames);
        newUnsupportedOpNames.addAll(other.unsupportedOpNames);

        int countUnique = newImportSupportedOpNames.size() + newUnsupportedOpNames.size();


        return new TFImportStatus(
                newModelPaths,
                newCantImportModelPaths,
                newReadErrorModelPaths,
                totalNumOps + other.totalNumOps,
                countUnique,
                newOpNames,
                newImportSupportedOpNames,
                newUnsupportedOpNames);
    }

}
