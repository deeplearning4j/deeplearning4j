package org.deeplearning4j.zoo;

import org.deeplearning4j.nn.api.Model;

/**
 * Interface for defining a model that can be instantiated and return
 * information about itself.
 */
public interface InstantiableModel {

    void setInputShape(int[][] inputShape);

    <M extends Model> M init();

    @Deprecated ModelMetaData metaData();

    Class<? extends Model> modelType();

    String pretrainedUrl(PretrainedType pretrainedType);

    long pretrainedChecksum(PretrainedType pretrainedType);

    String modelName();
}
