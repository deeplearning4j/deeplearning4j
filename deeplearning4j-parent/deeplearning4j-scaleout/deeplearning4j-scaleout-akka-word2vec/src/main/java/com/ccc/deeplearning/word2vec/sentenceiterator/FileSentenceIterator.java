package com.ccc.deeplearning.word2vec.sentenceiterator;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.LineIterator;

@SuppressWarnings("unchecked")
public class FileSentenceIterator extends BaseSentenceIterator {

	/*
	 * Used as a pair for when
	 * the number of sentences is not known
	 */
	private Iterator<File> fileIterator;
	private LineIterator currLineIterator;
	private File dir;

	public FileSentenceIterator(SentencePreProcessor preProcessor,File dir) {
		super(preProcessor);
		this.dir = dir;
		fileIterator = FileUtils.iterateFiles(dir, null, true);

	}

	public FileSentenceIterator(File dir) {
		this(null,dir);
	}


	@Override
	public String nextSentence() {
		if(currLineIterator != null) {
			if(currLineIterator.hasNext())  {
				String ret = currLineIterator.nextLine();	
				if(this.getPreProcessor() != null)
					ret =this.getPreProcessor().preProcess(ret);
				
				
				return ret;
			}

			else {
				nextLineIter();
				if(currLineIterator.hasNext()) {
					String ret = currLineIterator.nextLine();
					if(this.getPreProcessor() != null)
						ret =this.getPreProcessor().preProcess(ret);
					
					return ret;
				}


				else
					throw new IllegalStateException("No more lines found");
			}
		}
		else {
			nextLineIter();
			String ret =  currLineIterator.nextLine();
			if(this.getPreProcessor() != null)
				ret =this.getPreProcessor().preProcess(ret);
			
			return ret;
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
		return currLineIterator != null && currLineIterator.hasNext() || fileIterator.hasNext();
	}


	@Override
	public void reset() {
		fileIterator = FileUtils.iterateFiles(dir, null, true);


	}



}
