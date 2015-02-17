package org.deeplearning4j.cli.subcommands;

import org.deeplearning4j.cli.api.SubCommand;
import org.kohsuke.args4j.Option;

/**
 * Subcommand for model predictions
 *
 * Options:
 *      Required:
 *          --input: input data file for model
 *          --model: json configuration for model
 *          --output: destination for saving model
 *
 * Created by sonali on 2/10/15.
 */
public class Predict extends BaseSubCommand {

    @Option(name = "--input", usage = "input data",aliases = "-i", required = true)
    private String input = "input.txt";

    @Option(name = "--model", usage = "model for prediction", aliases = "-m", required = true)
    private String model = "model.json";

    @Option(name = "--output", usage = "location for saving model", aliases = "-o", required = true)
    private String output = "output.txt";

    public Predict(String[] args) {
        super(args);
    }


    @Override
    public void exec() {

    }

}
