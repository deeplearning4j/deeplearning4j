package org.deeplearning4j.text.tokenization.tokenizer;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class DefaultStreamTokenizer implements Tokenizer {

	private StreamTokenizer streamTokenizer;

	public DefaultStreamTokenizer(InputStream is) {
		Reader r = new BufferedReader(new InputStreamReader(is));
		streamTokenizer = new StreamTokenizer(r);
		char separator = ',';  
		streamTokenizer.resetSyntax();  
		// Characters other than comma and special characters are word characters  
		streamTokenizer.wordChars('\u0000', (char)(separator - 1));                 // Everything is a word character  
		streamTokenizer.wordChars((char)(separator + 1), '\u00ff');                 // except for the separator  
		streamTokenizer.whitespaceChars('\n', '\n');   
		// Make end-of-line whitespace(and therefore a word delimiter)  
		streamTokenizer.whitespaceChars(separator, separator);                      // Delimiter separates words  
		streamTokenizer.whitespaceChars(' ',' ');
		streamTokenizer.eolIsSignificant(true);                                     // End-of-line to be reported as TT_EOL  


	}

	@Override
	public boolean hasMoreTokens() {
		return streamTokenizer.ttype != StreamTokenizer.TT_EOF;
	}

	@Override
	public int countTokens() {
		return getTokens().size();
	}

	@Override
	public String nextToken() {
		StringBuffer sb = new StringBuffer();
		try {
			streamTokenizer.nextToken();
		} catch (IOException e1) {
			throw new RuntimeException(e1);
		}
		if(streamTokenizer.ttype == StreamTokenizer.TT_WORD) {
			sb.append(streamTokenizer.sval);
		} else if(streamTokenizer.ttype == StreamTokenizer.TT_NUMBER) {
			sb.append(streamTokenizer.nval);
		} else if(streamTokenizer.ttype == StreamTokenizer.TT_EOL) {
			try {
				while(streamTokenizer.ttype == StreamTokenizer.TT_EOL)
					streamTokenizer.nextToken();
			} catch (IOException e) {
				throw new RuntimeException(e);

			}
		}

		return sb.toString();
	}

	@Override
	public List<String> getTokens() {
		List<String> tokens = new ArrayList<String>();
		while(hasMoreTokens()) {
			tokens.add(nextToken());
		}
		return tokens;
	}

}
