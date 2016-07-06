package org.canova.cli.transforms;

import java.util.Collection;

import org.canova.api.writable.Writable;

public interface Transform {
	
	public void collectStatistics( Collection<Writable> vector );
	
	public void evaluateStatistics();
	
	public void transform( Collection<Writable> vector );

}
