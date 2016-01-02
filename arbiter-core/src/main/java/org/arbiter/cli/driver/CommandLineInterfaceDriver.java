/*
 *
 *  * Copyright 2016 Skymind,Inc.
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
package org.arbiter.cli.driver;


import org.arbiter.cli.subcommands.Evaluate;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.spi.SubCommand;
import org.kohsuke.args4j.spi.SubCommandHandler;
import org.kohsuke.args4j.spi.SubCommands;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CommandLineInterfaceDriver {
    private static final Logger log = LoggerFactory.getLogger(CommandLineInterfaceDriver.class);

    @Argument(required=true,index=0,metaVar="action",usage="subcommands, e.g., {evaluate}",handler=SubCommandHandler.class)
    @SubCommands({
            @SubCommand(name="evaluate",impl=Evaluate.class)
    })
    protected org.arbiter.cli.subcommands.SubCommand action;





    public void doMain(String[] args) throws Exception {
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
            action.execute();
        } catch( CmdLineException e ) {

            printUsage();

            //System.err.println(e.getMessage());
            return;
        }
    }

    public static void printUsage() {

        System.out.println( "Arbiter: Evaluation System" );
        System.out.println( "" );
        System.out.println( "\tUsage:" );
        System.out.println( "\t\tarbiter <command> <flags>" );
        System.out.println( "" );
        System.out.println( "\tCommands:" );
        System.out.println( "\t\tevaluate\t\tEvaluate DL4J Models with different measures (10-fold, ROC, F-1, etc...)" );
        System.out.println( "" );
        System.out.println( "\tExample:" );
        System.out.println( "\t\tarbiter evaluate -conf /tmp/iris_conf.txt " );

    }


    public static void main(String [] args) throws Exception {
        new CommandLineInterfaceDriver().doMain(args);

    }
}
