package com.ccc.deeplearning.word2vec.sentenceiterator;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;

@SuppressWarnings("unchecked")
public class FileSentenceIterator implements SentenceIterator {

	/*
	 * Used as a pair for when
	 * the number of sentences is not known
	 */
	private Iterator<File> fileIterator;
	private LineIterator currLineIterator;
	private File dir;
	
	public FileSentenceIterator(File dir) {
		this.dir = dir;
		fileIterator = FileUtils.iterateFiles(dir, null, true);
		
	}
	
	
	@Override
	public String nextSentence() {
		if(currLineIterator != null) {
			if(currLineIterator.hasNext()) 
		         return currLineIterator.nextLine();		
			
			else {
				nextLineIter();
				if(currLineIterator.hasNext()) 
					return currLineIterator.nextLine();
				
				else
					throw new IllegalStateException("No more lines found");
			}
		}
		else {
			nextLineIter();
			return currLineIterator.nextLine();	
		}
		
	}


	
	private void nextLineIter() {
		if(fileIterator.hasNext()) {
			try {
				currLineIterator = FileUtils.lineIterator(fileIterator.next());
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
	}
	
	@Override
	public boolean hasNext() {
		return currLineIterator != null && currLineIterator.hasNext() && fileIterator.hasNext();
	}


	@Override
	public void reset() {
		fileIterator = FileUtils.iterateFiles(dir, null, true);

		
	}

	

}
