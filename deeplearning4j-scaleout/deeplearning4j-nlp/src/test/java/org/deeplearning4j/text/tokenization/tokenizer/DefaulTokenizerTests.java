package org.deeplearning4j.text.tokenization.tokenizer;

import static org.junit.Assert.*;

import java.io.ByteArrayInputStream;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

public class DefaulTokenizerTests {
	
	
	@Test
	public void testDefaultTokenizer() throws Exception {
		String toTokenize = "Mary had a little lamb.";
		TokenizerFactory t = new DefaultTokenizerFactory();
		Tokenizer tokenizer = t.create(toTokenize);
		Tokenizer tokenizer2 = t.create(new ByteArrayInputStream(toTokenize.getBytes()));
		while(tokenizer.hasMoreTokens()) {
		   assertEquals(tokenizer.nextToken(),tokenizer2.nextToken());
		}
		
		
		ClassPathResource resource = new ClassPathResource("/reuters/5250");
		String str = FileUtils.readFileToString(resource.getFile());
		int stringCount = t.create(str).countTokens();
		int stringCount2 = t.create(resource.getInputStream()).countTokens();
		assertTrue(Math.abs(stringCount - stringCount2) < 2);
		
	}
	
	

}
