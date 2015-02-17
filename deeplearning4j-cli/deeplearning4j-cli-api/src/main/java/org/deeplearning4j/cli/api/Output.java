package org.deeplearning4j.cli.api;

/**
 * Interface for saving the model
 * Created by sonali on 2/10/15.
 */
public interface Output extends SubCommand {

    void saveModel();
}
