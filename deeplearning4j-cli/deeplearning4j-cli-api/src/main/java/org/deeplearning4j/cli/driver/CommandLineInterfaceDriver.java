package org.deeplearning4j.cli.driver;

import java.util.Arrays;

import org.deeplearning4j.cli.subcommands.Train;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CLI Driver for dl4j.
 *
 * Supports the train command
 *
 * @author Adam Gibson
 */
public class CommandLineInterfaceDriver {

	private static Logger log = LoggerFactory.getLogger(CommandLineInterfaceDriver.class);

    /**
     * Print the usage for the command.
     */
	public static void printUsage() {
        log.info( "Usage: " );
		log.info( "\tdl4j [command] [params] " );
		log.info( "Commands: " );
		log.info( "\ttrain\tbuild a deep learning model " );
		log.info( "\ttest\ttest a deep learning model " );
		log.info( "\tpredict\tscore new records against a deep learning model " );
		log.info( "" );

	}

	public static void main(String [ ] args) {

		if ( args.length < 1 )
			printUsage();


		else if ("train".equals( args[0])) {

			String[] vec_params = Arrays.copyOfRange(args, 1, args.length);

			Train train = new Train(vec_params);
			train.exec();
			log.info("[DONE] - Test Mode");
		}
		else

			printUsage();






	}

}
