package org.deeplearning4j.cli.subcommands;

import org.kohsuke.args4j.Option;

/**
 * Subcommand for testing model
 *
 * Options:
 *      Required:
 *          --input: input data file for model
 *          --model: json configuration for model
 *
 * @author sonali
 */
public class Test extends BaseSubCommand {

    @Option(name = "--input", usage = "input data",aliases = "-i", required = true)
    private String input = "input.txt";

    @Option(name = "--model", usage = "model for prediction", aliases = "-m", required = true)
    private String model = "model.json";

    public Test(String[] args) {
        super(args);
    }

    @Override
    public void exec() {

    }
}
