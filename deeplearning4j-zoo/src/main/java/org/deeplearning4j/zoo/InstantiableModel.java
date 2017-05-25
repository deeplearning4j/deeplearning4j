package org.deeplearning4j.zoo;

import org.deeplearning4j.nn.api.Model;

/**
 * Interface for defining a model that can be instantiated and return
 * information about itself.
 */
public interface InstantiableModel {

    public Model init();

    public ModelMetaData metaData();

    public ZooType zooType();

    public Class<? extends Model> modelType();

    public String pretrainedImageNetUrl();
}
