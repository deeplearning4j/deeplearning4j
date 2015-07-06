package org.arbiter.cli.driver;

import static org.junit.Assert.*;

import org.junit.Test;

public class TestCommandLineInterfaceDriver {

	@Test
	public void testMainCLIDriverEntryPoint_NoArgs() throws Exception {

		String[] args = {  };

		CommandLineInterfaceDriver.main( args );

	}

}
