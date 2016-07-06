package org.canova.cli.shuffle;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.ListIterator;
import java.util.Random;

import org.canova.api.writable.Writable;

/**
 * Record Ordering Shuffler
 * 
 * Just a good old fashioned way to shuffle the output of the records
 * 
 * Yes, it is memory based (and bound) for now. (We know. We Know.)
 * 
 * 
 * 
 * @author Josh Patterson
 *
 */
public class Shuffler {

	public List< Collection<Writable> > records;// =new List<String>();
	//public int numRecords = -1;
	private Random rand = new Random();

	private ListIterator< Collection<Writable> > iterator = null;
	
	/**
	 * We could probably infer this, but I'm lazy
	 * 
	 */
	public Shuffler() {
		
		//this.numRecords = numberRecords;
		this.records = new ArrayList<>();
		
	}
	
	public void addRecord( Collection<Writable> record ) {
		
		int slot = this.getRandomSlot();
		
		// handles moving other items around on collisions
		this.records.add( slot, record );
		
	}
	
	private int getRandomSlot() {

		int min = 0;
		

	    // nextInt is normally exclusive of the top value,
	    // so add 1 to make it inclusive

		return rand.nextInt(((this.records.size()) - min) + 1) + min;
		
	}
	
	
	public boolean hasNext() {

		// cant touch it til we read, or ConcurrentModificatoinException
		if ( null == this.iterator ) {
			this.iterator = this.records.listIterator();
		}
		
		return this.iterator.hasNext();
	}
	
	public Collection<Writable> next() {
		
		// cant touch it til we read, or ConcurrentModificationException
		if ( null == this.iterator ) {
			this.iterator = this.records.listIterator();
		}
				
		return this.iterator.next();
	}
	
	/*
	for(ar=0;ar<q.size();ar++){

	int ran= random(1,q.size());
	*/
}
