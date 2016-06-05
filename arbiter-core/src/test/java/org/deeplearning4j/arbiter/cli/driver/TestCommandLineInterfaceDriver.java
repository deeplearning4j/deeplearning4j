package org.deeplearning4j.arbiter.cli.driver;

import org.junit.Test;

public class TestCommandLineInterfaceDriver {

	@Test
	public void testMainCLIDriverEntryPoint_NoArgs() throws Exception {

		String[] args = {  };

		CommandLineInterfaceDriver.main( args );

	}

}
