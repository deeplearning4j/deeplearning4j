package org.deeplearning4j.zoo;

import org.deeplearning4j.nn.api.Model;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * A zoo model is instantiable, returns information about itself, and can download
 * pretrained models if available.
 */
public abstract class ZooModel implements InstantiableModel {

    public boolean pretrainedAvailable() {
        return false;
    }

    public Model getPretrained() {
        String modelFile = modelType().toString().toLowerCase()+"."+getLatestRevision()+".zip";

        // Depends on URL scheme not yet implemented
        throw new NotImplementedException();
    }

    public int getLatestRevision() {
        // Depends on URL scheme not yet implemented
        throw new NotImplementedException();
    }
}
