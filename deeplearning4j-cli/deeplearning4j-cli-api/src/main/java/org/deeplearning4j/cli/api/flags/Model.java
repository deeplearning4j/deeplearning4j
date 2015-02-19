package org.deeplearning4j.cli.api.flags;

import org.deeplearning4j.cli.subcommands.SubCommand;

/**
 * Model flag for setting model configurations
 *
 * @author sonali
 */
public class Model implements Flag {
    /**
     * JSON model configuration passed in
     *
     */
    @Override
    public <E> E value(String value) throws Exception {
        return null;
    }

}
