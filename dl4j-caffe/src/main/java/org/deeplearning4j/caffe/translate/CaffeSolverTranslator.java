package org.deeplearning4j.caffe.translate;

import org.apache.commons.lang3.text.WordUtils;
import org.deeplearning4j.caffe.proto.Caffe;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder;
import org.deeplearning4j.caffe.common.NNConfigBuilderContainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author jeffreytang
 */
public class CaffeSolverTranslator implements CaffeSpecialTranslator {

    protected static Logger log = LoggerFactory.getLogger(CaffeSolverTranslator.class);

    public CaffeSolverTranslator() {
        fillSolverParamMappings();
    }

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

    @Override
    public void specialTranslation(String caffeFieldName, Object caffeFieldValue, String builderFieldName,
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

    public void translate(Caffe.SolverParameter solver, NNConfigBuilderContainer builderContainer)
            throws IllegalAccessException, NoSuchFieldException {

        // The builder should not already be assigned since solver should be the fist one parsed.
        // Instantiate new builder later
        if (builderContainer.getBuilder() != null)
            throw new IllegalStateException("Builder should not already be assigned. " +
                    "Parse Caffe solver first before Net.");

        // Get the map of map (SolverFieldName -> BuilderFieldName -> SolverFieldValue)
        Map<String, Map<String, Object>> map = CaffeTranslatorUtils.caffeField2builderField2caffeVal(solver, solverParamMappings);
        // Turn the map to a list of list
        List<List<Object>> lst = CaffeTranslatorUtils.caffeFieldBuilderFieldCaffeValIter(map);

        // translate lst to (BuilderFieldName -> BuilderFieldValue)
        Map<String, Object> builderParamMap = new HashMap<>();
        CaffeTranslatorUtils.translation2BuilderFieldBuilderValMap(lst, builderParamMap, this);

        // Use the map to set value to the relevant fields in the builder
        Builder builder = new NeuralNetConfiguration.Builder();
        CaffeTranslatorUtils.applyMapToBuilder(builder, builderParamMap);

        // Put Builder into BuilderContainer
        builderContainer.setBuilder(builder);
    }
}
