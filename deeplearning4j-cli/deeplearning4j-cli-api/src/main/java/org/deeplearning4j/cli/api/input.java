package org.deeplearning4j.cli.api;

import java.net.URI;

/**
 * Interface for loading input data for the model
 *
 * Created by sonali on 2/10/15.
 */
public interface Input extends SubCommand {

    URI parseUri();

    void process();

}
