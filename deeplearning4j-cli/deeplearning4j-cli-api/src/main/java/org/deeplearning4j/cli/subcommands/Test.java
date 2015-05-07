/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

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

    @Option(name = "--runtime", usage = "runtime- local, Hadoop, Spark, etc.", aliases = "-r", required = false)
    private String runtime = "local";

    @Option(name = "--pcdroperties", usage = "configuration for distributed systems", aliases = "-p", required = false)
    private String properties;

    public Test(String[] args) {
        super(args);
    }

    @Override
    public void exec() {

    }
}
