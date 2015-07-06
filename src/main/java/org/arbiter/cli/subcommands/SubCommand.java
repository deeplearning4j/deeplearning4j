package org.arbiter.cli.subcommands;

/**
 * A subcommand used for handling input
 *
 * @author Adam Gibson
 */
public interface SubCommand {

    /**
     * Execute the input
     */
    void execute() throws Exception;

}
