package com.ccc.deeplearning.word2vec.sentenceiterator;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;

/**
 * Each line is a sentence
 * @author Adam Gibson
 *
 */
public class LineSentenceIterator extends BaseSentenceIterator {

	
	private File file;
	private LineIterator iter;
	
	
	public LineSentenceIterator(File f) {
		if(!f.exists() || !f.isFile())
			throw new IllegalArgumentException("Please specify an existing file");
		this.file = f;
		try {
			iter = FileUtils.lineIterator(file);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		
	}
	
	
	
	@Override
	public String nextSentence() {
		return iter.nextLine();
	}

	@Override
	public boolean hasNext() {
		return iter.hasNext();
	}

	@Override
	public void reset() {
		try {
			if(iter != null)
				iter.close();
			iter = FileUtils.lineIterator(file);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		
	}

	

}
