package org.deeplearning4j.cli.subcommands;

import org.kohsuke.args4j.Option;

/**
 * Created by sonali on 2/11/15.
 */
public class DummySubCommand extends BaseSubCommand {

    @Option(name = "--input", usage = "input data",aliases = "-i", required = true)
    private String dummyValue;

    /**
     * @param args arguments for command
     */
    public DummySubCommand(String[] args) {
        super(args);
    }

    @Override
    public void exec() {

    }

    public String getDummyValue() {
        return dummyValue;
    }

    public void setDummyValue(String dummyValue) {
        this.dummyValue = dummyValue;
    }
}
