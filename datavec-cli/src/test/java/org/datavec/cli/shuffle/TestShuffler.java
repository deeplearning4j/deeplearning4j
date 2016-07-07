/*
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.cli.shuffle;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Collection;

import org.datavec.api.io.data.Text;
import org.datavec.api.writable.Writable;
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
