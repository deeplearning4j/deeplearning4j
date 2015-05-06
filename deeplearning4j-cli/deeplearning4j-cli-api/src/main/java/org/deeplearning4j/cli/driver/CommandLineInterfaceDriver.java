package org.deeplearning4j.cli.driver;

import java.util.Arrays;

import org.deeplearning4j.cli.subcommands.Train;



public class CommandLineInterfaceDriver {

	
	public static void printUsage() {
		
		System.out.println( "Usage: " );
		System.out.println( "\tdl4j [command] [params] " );
		System.out.println( "Commands: " );
		System.out.println( "\ttrain\tbuild a deep learning model " );
		System.out.println( "\ttest\ttest a deep learning model " );
		System.out.println( "\tpredict\tscore new records against a deep learning model " );
		System.out.println( "" );
		
	}

	public static void main(String [ ] args) {
	    /*
		System.out.println( "CommandLineInterfaceDriver > Printing args:");
		
		for ( String arg : args ) {
			
			System.out.println( ">> " + arg );
			
		}
		*/
		
		if ( args.length < 1 ) {
		
			printUsage();
			
		} else if ("train".equals( args[ 0 ] )) {

			String[] vec_params = Arrays.copyOfRange(args, 1, args.length);
			/*
			Vectorize vecCommand = new Vectorize( vec_params );
			try {
				vecCommand.execute();
			} catch (CanovaException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			*/
			
			String[] cmd = {
	                "--input", "iris.txt", "--model", "model.json", "--output", "test_output.txt"
	        };

	        Train train = new Train(cmd);
	        
	        System.out.println( "[DONE] - Test Mode" ); 
			

		} else if ("test".equals( args[ 0 ] )) {
			
		} else if ("predict".equals( args[ 0 ] )) {
			
			
		} else {
			
			//System.out.println( "Canova's command line system only supports the 'vectorize' command." );
			printUsage();
			
		}
		
		
		
		
	}		
	
}
