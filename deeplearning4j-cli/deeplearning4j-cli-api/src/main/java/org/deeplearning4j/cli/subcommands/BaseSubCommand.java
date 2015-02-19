package org.deeplearning4j.cli.subcommands;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Base class for subcommand
 *
 * @author sonali
 */
public abstract class BaseSubCommand implements SubCommand {
    protected String[] args;
    private static Logger log = LoggerFactory.getLogger(BaseSubCommand.class);

    /**
     *
     * @param args arguments for command
     */
    public BaseSubCommand(String[] args) {
        this.args = args;
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            parser.printUsage(System.err);
            log.error("Unable to parse args",e);
        }

    }
}
