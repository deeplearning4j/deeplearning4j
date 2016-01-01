package org.arbiter.cli.subcommands;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Enumeration;
import java.util.Properties;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Evaluate implements SubCommand {

    private static final Logger log = LoggerFactory.getLogger(Evaluate.class);

    public static final String MODEL_PATH_KEY = "arbiter.model.path";

    protected String[] args;

    public boolean validCommandLineParameters = true;

    @Option(name = "-conf", usage = "Sets a configuration file to drive the evaluation process")
    public String configurationFile = "";

    public Properties configProps = null;    
    
    
    public Evaluate() {
    }

    public Evaluate(String[] args) {
    	
        this.args = args;
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            this.validCommandLineParameters = false;
            parser.printUsage(System.err);
            log.error("Unable to parse args", e);
        }
    	
    	
    }
        
    
    
    public void loadConfigFile() throws IOException {

        this.configProps = new Properties();

        //Properties prop = new Properties();
        try (InputStream in = new FileInputStream(this.configurationFile)) {
            this.configProps.load(in);
        }

    }

    /**
     * Dont change print stuff, its part of application console output UI
     * 
     */
    public void debugLoadedConfProperties() {
        Properties props = this.configProps; //System.getProperties();
        Enumeration e = props.propertyNames();

        System.out.println("\n--- Start Canova Configuration ---");

        while (e.hasMoreElements()) {
            String key = (String) e.nextElement();
            System.out.println(key + " -- " + props.getProperty(key));
        }

        System.out.println("---End Canova Configuration ---\n");
    }
    
    public static void printUsage() {
    	
    	System.out.println( "Arbiter: Model Evaluation Engine" );
    	System.out.println( "" );
    	System.out.println( "\tUsage:" );
    	System.out.println( "\t\tarbiter evaluation -conf <conf_file>" );
    	System.out.println( "" );
    	System.out.println( "\tConfiguration File:" );
    	System.out.println( "\t\tContains a list of property entries that describe the model evaluation process" );
    	System.out.println( "" );
    	System.out.println( "\tExample:" );
    	System.out.println( "\t\tarbiter evaluate -conf /tmp/iris_conf.txt " );
    	
    	
    }    
    
	@Override
	public void execute() throws Exception {
		// TODO Auto-generated method stub
		
	}
    
    
    
}
