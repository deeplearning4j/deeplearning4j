package org.nd4j.serde.json;


import lombok.NonNull;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.*;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.util.HashMap;
import java.util.Map;

/**
 * Deserializer for ILossFunction JSON in legacy format - see {@link BaseLegacyDeserializer}
 *
 * @author Alex Black
 */
public class LegacyILossFunctionDeserializer extends BaseLegacyDeserializer<ILossFunction> {
    private static final Map<String,String> LEGACY_NAMES = new HashMap<>();

    private static ObjectMapper legacyMapper;

    public static void setLegacyJsonMapper(ObjectMapper mapper){
        legacyMapper = mapper;
    }

    static {
        LEGACY_NAMES.put("BinaryXENT", LossBinaryXENT.class.getName());
        LEGACY_NAMES.put("CosineProximity", LossCosineProximity.class.getName());
        LEGACY_NAMES.put("Hinge", LossHinge.class.getName());
        LEGACY_NAMES.put("KLD", LossKLD.class.getName());
        LEGACY_NAMES.put("MAE", LossMAE.class.getName());
        LEGACY_NAMES.put("L1", LossL1.class.getName());
        LEGACY_NAMES.put("MAPE", LossMAPE.class.getName());
        LEGACY_NAMES.put("MCXENT", LossMCXENT.class.getName());
        LEGACY_NAMES.put("MSE", LossMSE.class.getName());
        LEGACY_NAMES.put("L2", LossL2.class.getName());
        LEGACY_NAMES.put("MSLE", LossMSLE.class.getName());
        LEGACY_NAMES.put("NegativeLogLikelihood", LossNegativeLogLikelihood.class.getName());
        LEGACY_NAMES.put("Poisson", LossPoisson.class.getName());
        LEGACY_NAMES.put("SquaredHinge", LossSquaredHinge.class.getName());
        LEGACY_NAMES.put("MultiLabel", LossMultiLabel.class.getName());
        LEGACY_NAMES.put("FMeasure", LossFMeasure.class.getName());

        //The following didn't previously have subtype annotations - hence will be using default name (class simple name)
        LEGACY_NAMES.put(LossMixtureDensity.class.getSimpleName(), LossMixtureDensity.class.getName());
    }


    @Override
    public Map<String, String> getLegacyNamesMap() {
        return LEGACY_NAMES;
    }

    @Override
    public ObjectMapper getLegacyJsonMapper() {
        return legacyMapper;
    }

    @Override
    public Class<?> getDeserializedType() {
        return ILossFunction.class;
    }

    public static void registerLegacyClassDefaultName(@NonNull Class<? extends ILossFunction> clazz){
        registerLegacyClassSpecifiedName(clazz.getSimpleName(), clazz);
    }

    public static void registerLegacyClassSpecifiedName(@NonNull String name, @NonNull Class<? extends ILossFunction> clazz){
        LEGACY_NAMES.put(name, clazz.getName());
    }
}
