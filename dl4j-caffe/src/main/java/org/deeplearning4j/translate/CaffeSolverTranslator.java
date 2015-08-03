package org.deeplearning4j.translate;

import org.apache.commons.lang3.text.WordUtils;
import org.deeplearning4j.caffe.Caffe.SolverParameter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * @author jeffreytang
 */
public class CaffeSolverTranslator {

    protected static Logger log = LoggerFactory.getLogger(CaffeSolverTranslator.class);

    public Map<String, String> solverParamMappings;
    private void fillSolverParamMappings() {
        solverParamMappings = new HashMap<String, String>() {{
            put("baseLr_", "lr");
            put("momentum_", "momentum");
            put("maxIter_", "numIterations");
            put("randomSeed_", "seed");
            put("solverType_", "optimizationAlgo");
            put("regularizationType_", "useRegularization");
            put("display_", "");
            put("lrPolicy_", "");
            put("gamma_", "");
            put("power_", "");
            put("weightDecay_", "");
            put("stepsize_", "");
            put("stepvalue_", "");
            put("clipGradients_", "");
            put("solverMode_", "");
            put("deviceId_", "");
            put("delta_", "");
        }};
    }

    private void specialTranslation(String caffeFieldName, Object caffeFieldValue, String builderFieldName,
                                   Map<String, Object> builderParamMap) {

        //If the field value is a string, lower case it
        if (caffeFieldValue instanceof String)
            caffeFieldValue = WordUtils.capitalizeFully((String) caffeFieldValue);

        // Process solverType / OptimizationAlgo
        if (caffeFieldName.equals("solverType_")) {
            if (caffeFieldValue.equals("sgd")) {
                builderParamMap.put(builderFieldName, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
            } else {
                log.info("Only SGD is supported. Switch to SGD.");
                builderParamMap.put(builderFieldName, OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
            }
        // Process regularizationType
        } else if (caffeFieldName.equals("regularizationType_")) {
            if (caffeFieldValue.equals("l1")) {
                builderParamMap.put("useRegularization", true);
                builderParamMap.put("l1", true);
            } else if (caffeFieldValue.equals("l2")) {
                builderParamMap.put("useRegularization", true);
                builderParamMap.put("l2", true);
            } else {
                builderParamMap.put("useRegularization", false);
            }
        }

        // Minimize is always true in Caffe
        builderParamMap.put("minimize", true);
    }

    public CaffeSolverTranslator() {
        fillSolverParamMappings();
    }

    public NNCofigBuilderContainer translate(SolverParameter solver,
                                             NNCofigBuilderContainer builderContainer)
            throws IllegalAccessException, NoSuchFieldException {

        // The builder should not already be assigned since solver should be the fist one parsed.
        // Instantiate new builder later
        if (builderContainer.getBuilder() != null)
            throw new IllegalStateException("Builder should not already be assigned. " +
                    "Parse Caffe solver first before Net.");

        // Get the map with the BuilderFieldName, SolverFieldName and SolverFieldValue
        // Get a Map of Map (SolverFieldName mapped to BuilderFieldName mapped to SolverFieldValue)
        // Use reflection to loop through all fields in solver and map to current HashMap
        Map<String, Map<String, Object>> solverBuilderParamMap =
                CaffeTranslatorUtils.getFieldValueList(solver, solverParamMappings);

        // Get the map with the BuilderFieldName to the adjusted BuilderFieldValue
        // BuilderParamMap maps the builderFieldName(String) to buildFieldValue(Object)
        Map<String, Object> builderParamMap = new HashMap<>();
        for (Map.Entry<String, Map<String, Object>> entry : solverBuilderParamMap.entrySet()) {
            String solverFieldName = entry.getKey();
            Map.Entry builderFieldName2SolverValue = entry.getValue().entrySet().iterator().next();
            String builderFieldName = (String)builderFieldName2SolverValue.getKey();
            Object solverFieldValue = builderFieldName2SolverValue.getValue();
            // Translations to fill the
            specialTranslation(solverFieldName, solverFieldValue, builderFieldName, builderParamMap);
            CaffeTranslatorUtils.regularTranslation(solverFieldValue, builderFieldName, builderParamMap);
        }

        // Set fields in Builder using the last map
        // Instantiate new builder
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        CaffeTranslatorUtils.setFieldFromMap(builder, builderParamMap);

        // Put the builder in the builderContainer
        builderContainer.setBuilder(builder);

        return builderContainer;
    }
}
