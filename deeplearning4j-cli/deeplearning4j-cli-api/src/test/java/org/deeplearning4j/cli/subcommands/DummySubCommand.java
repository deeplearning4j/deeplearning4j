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
    public void execute() {

    }

    public String getDummyValue() {
        return dummyValue;
    }

    public void setDummyValue(String dummyValue) {
        this.dummyValue = dummyValue;
    }
}
