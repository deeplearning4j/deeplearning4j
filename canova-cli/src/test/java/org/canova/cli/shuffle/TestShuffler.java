package org.canova.cli.shuffle;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Collection;

import org.canova.api.io.data.Text;
import org.canova.api.writable.Writable;
import org.junit.Test;

public class TestShuffler {

	@Test
	public void test() {
		
		Shuffler shuffle = new Shuffler();
		
		Collection<Writable> vector1 = new ArrayList<>();
		vector1.add(new Text("Go Dogs, Go") );
		vector1.add(new Text("label_A") );

		Collection<Writable> vector2 = new ArrayList<>();
		vector2.add(new Text("Go Dogs, Go 2") );
		vector2.add(new Text("label_B") );

		Collection<Writable> vector3 = new ArrayList<>();
		vector3.add(new Text("Go Dogs, Go 3") );
		vector3.add(new Text("label_C") );
		
		shuffle.addRecord(vector1);
		shuffle.addRecord(vector2);
		shuffle.addRecord(vector3);
		
		assertEquals(3, shuffle.records.size());
		
		while (shuffle.hasNext()) {
			
			Collection<Writable> s = shuffle.next();
			System.out.println( s );
			
		}
		
		
	}

}
