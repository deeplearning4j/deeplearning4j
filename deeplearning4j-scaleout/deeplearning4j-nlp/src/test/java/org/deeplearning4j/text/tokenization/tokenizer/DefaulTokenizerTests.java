package org.deeplearning4j.text.tokenization.tokenizer;

import static org.junit.Assert.*;

import java.io.ByteArrayInputStream;

import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;

public class DefaulTokenizerTests {
	
	
	@Test
	public void testDefaultTokenizer() {
		String toTokenize = "Mary had a little lamb.";
		TokenizerFactory t = new DefaultTokenizerFactory();
		Tokenizer tokenizer = t.create(toTokenize);
		Tokenizer tokenizer2 = t.create(new ByteArrayInputStream(toTokenize.getBytes()));
		while(tokenizer.hasMoreTokens()) {
		   assertEquals(tokenizer.nextToken(),tokenizer2.nextToken());
		}
	}
	
	

}
