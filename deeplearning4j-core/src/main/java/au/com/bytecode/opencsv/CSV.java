package au.com.bytecode.opencsv;

/**
Copyright 2005 Bytecode Pty Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

import static au.com.bytecode.opencsv.CSVParser.DEFAULT_ESCAPE_CHARACTER;
import static au.com.bytecode.opencsv.CSVParser.DEFAULT_IGNORE_LEADING_WHITESPACE;
import static au.com.bytecode.opencsv.CSVParser.DEFAULT_QUOTE_CHARACTER;
import static au.com.bytecode.opencsv.CSVParser.DEFAULT_SEPARATOR;
import static au.com.bytecode.opencsv.CSVParser.DEFAULT_STRICT_QUOTES;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.Charset;


/**
 * A very simple CSV format container. It allows to construct {@link CSVReader} and {@link CSVWriter} objects. 
 * Also it provides helper methods to read and write CSV. Example: <pre> {@code
 *    
 *    // define CSV format 
 *    CSV csv = CSV
 *          .separator(',')
 *          .skipLines(1)
 *          .create();
 *		
 *    // write CSV file
 *    csv.write("example.csv", new CSVWriteProc() {
 *        public void process(CSVWriter out) {
 *            out.writeNext("Col1", "Col2");
 *            out.writeNext("Val1", "Val2");
 *        }
 *    });
 *		
 *    // read CSV file
 *    csv.read("example.csv", new CSVReadProc() {
 *        public void procRow(int rowIndex, String... values) {
 *            System.out.println(Arrays.asList(values));
 *        }
 *    });}</pre>
 *
 * @author Dmitry Sanatin
 */
public class CSV {
	
	// Common
    private final char separator;
    private final char quotechar;
    private final char escapechar;
    private final Charset charset;
    
    // Writer
    private final String lineEnd;
    
    // Reader
    private final int skipLines;    
    private final boolean strictQuotes;
    private final boolean ignoreLeadingWhiteSpace;


	private CSV(char separator, char quotechar, char escapechar, String lineEnd, int skipLines,
			boolean strictQuotes, boolean ignoreLeadingWhiteSpace, Charset charset) {
		this.separator = separator;
		this.quotechar = quotechar;
		this.escapechar = escapechar;
		this.lineEnd = lineEnd;
		this.skipLines = skipLines;
		this.strictQuotes = strictQuotes;
		this.ignoreLeadingWhiteSpace = ignoreLeadingWhiteSpace;
		this.charset = charset;
	}

	private CSV() {
		this(	DEFAULT_SEPARATOR, 
				DEFAULT_QUOTE_CHARACTER, 
				DEFAULT_ESCAPE_CHARACTER, 
				CSVWriter.DEFAULT_LINE_END, 
				CSVReader.DEFAULT_SKIP_LINES,
				DEFAULT_STRICT_QUOTES,
				DEFAULT_IGNORE_LEADING_WHITESPACE,
				Charset.defaultCharset());
	}
	
    /**
     * Constructs <code>CSVWriter</code> using configured CSV format.
     *
     * @param writer
     *            the writer to an underlying CSV.
     */
	public CSVWriter writer(Writer writer) {
		return new CSVWriter(writer, separator, quotechar, escapechar, lineEnd);
	}
	
	
	/**
	 * Constructs <code>CSVWriter</code> using configured CSV file format.
	 * 
	 * @param os 
	 *            the output stream to an underlying CSV.
	 */
	public CSVWriter writer(OutputStream os) {
		return writer(new OutputStreamWriter(os, charset));
	}

	/**
	 * Constructs <code>CSVWriter</code> using configured CSV file format.
	 * 
	 * @param file 
	 *            CSV file
	 */
	public CSVWriter writer(File file) {
		try {
			return writer(new FileOutputStream(file));
		} catch (FileNotFoundException e) {
			throw new CSVRuntimeException(e);
		}
	}
	
	/**
	 * Constructs <code>CSVWriter</code> using configured CSV file format.
	 * 
	 * @param fileName 
	 * 					name of CSV file
	 */
	public CSVWriter writer(String fileName) {
		return writer(new File(fileName));
	}	

	
	/**
	 * Write CSV using the supplied {@link CSVWriteProc}.
	 * 
	 * @param writer 
	 * 				the writer to an underlying CSV. The writer will not be closed.
	 * @param proc 
	 * 				the {@link CSVWriteProc} to use for CSV writing
	 */
	public void write(Writer writer, CSVWriteProc proc) {
		write(writer(writer), proc);
	}

	/**
	 * Write CSV using the supplied {@link CSVWriteProc}.
	 * 
	 * @param os 
	 *            the output stream to an underlying CSV. The output stream will not be closed.
	 * @param proc 
	 * 				the {@link CSVWriteProc} to use for CSV writing
	 */
	public void write(OutputStream os, CSVWriteProc proc) {
		write(writer(os), proc);
	}
	
	/**
	 * Write CSV using the supplied {@link CSVWriteProc}.
	 * 
	 * @param fileName 
	 *            	the name of CSV file
	 * @param proc 
	 * 				the {@link CSVWriteProc} to use for CSV writing
	 */
	public void write(String fileName, CSVWriteProc proc) {
		write(new File(fileName), proc);
	}

	
	/**
	 * Write CSV using the supplied {@link CSVWriteProc}.
	 * 
	 * @param writer 
	 * 				the writer to an underlying CSV. The writer will not be closed.
	 * @param proc 
	 * 				the {@link CSVWriteProc} to use for CSV writing
	 */
	public void write(CSVWriter writer, CSVWriteProc proc) {
		try {
			writer.write(proc);
			writer.flush();
		} catch (IOException e) {
			throw new CSVRuntimeException(e);
		}
	}
	
	/**
	 * Write CSV using the supplied {@link CSVWriteProc}.
	 * 
	 * @param file 
	 *            CSV file
	 * @param proc 
	 * 				the {@link CSVWriteProc} to use for CSV writing
	 */	
	public void write(File file, CSVWriteProc proc) {
		writeAndClose(writer(file), proc);
	}
	
	/**
	 * Write CSV using the supplied {@link CSVWriteProc}.
	 * 
	 * @param writer 
	 * 				the writer to an underlying CSV. The writer will be closed.
	 * @param proc 
	 * 				the {@link CSVWriteProc} to use for CSV writing
	 */
	public void writeAndClose(Writer writer, CSVWriteProc proc) {
		writeAndClose(writer(writer), proc);
	}

	
	/**
	 * Write CSV using the supplied {@link CSVWriteProc}.
	 * 
	 * @param os 
	 *            the output stream to an underlying CSV. The output stream will be closed.
	 * @param proc 
	 * 				the {@link CSVWriteProc} to use for CSV writing
	 */	
	public void writeAndClose(OutputStream os, CSVWriteProc proc) {
		writeAndClose(writer(os), proc);
	}
	
	/**
	 * Write CSV using the supplied {@link CSVWriteProc}.
	 * 
	 * @param writer 
	 * 				the writer to an underlying CSV. The writer will be closed.
	 * @param proc 
	 * 				the {@link CSVWriteProc} to use for CSV writing
	 */
	public void writeAndClose(CSVWriter writer, CSVWriteProc proc) {
		try {
			writer.write(proc);
		} catch ( RuntimeException re )  {
			try {
				writer.close();
			} catch ( Exception e ) {}
			throw re;
		} 
		
		try {
			writer.close();
		} catch ( Exception e ) {
			throw new CSVRuntimeException(e);
		}
	}
	
	
    /**
     * Constructs <code>CSVReader</code> using configured CSV format.
     *
     * @param reader
     *            the reader of an underlying CSV.
     */
	public CSVReader reader(Reader reader) {
		return new CSVReader(reader, separator, quotechar, escapechar, skipLines, 
				strictQuotes, ignoreLeadingWhiteSpace);
	}
	
    /**
     * Constructs <code>CSVReader</code> using configured CSV format.
     *
     * @param is
     *            the input stream of an underlying CSV.
     */
	public CSVReader reader(InputStream is) {
		return reader(new InputStreamReader(is, charset));
	}

    /**
     * Constructs <code>CSVReader</code> using configured CSV format.
     *
     * @param file
     *            the CSV file
     */
	public CSVReader reader(File file) {
		try {
			return reader(new FileInputStream(file));
		} catch (IOException e) {
			throw new CSVRuntimeException(e);
		}
	}
	
    /**
     * Constructs <code>CSVReader</code> using configured CSV format.
     *
     * @param fileName
     *            the file name.
     */
	public CSVReader reader(String fileName) {
		return reader(new File(fileName));
	}
	
	/**
	 * Read CSV using the supplied {@link CSVReadProc}.
	 * 
	 * @param is 
	 * 				the input stream of an underlying CSV. The input stream will not be closed.
	 * @param proc 
	 * 				the {@link CSVReadProc} to use for CSV reading
	 */	
	public void read(InputStream is, CSVReadProc proc) {
		read(reader(is), proc);
	}
	
	/**
	 * Read CSV using the supplied {@link CSVReadProc}.
	 * 
	 * @param reader 
	 * 				the reader of an underlying CSV. The reader will not be closed.
	 * @param proc 
	 * 				the {@link CSVReadProc} to use for CSV reading
	 */	
	public void read(Reader reader, CSVReadProc proc) {
		read(reader(reader), proc);
	}
	
	/**
	 * Read CSV using the supplied {@link CSVReadProc}.
	 * 
	 * @param file 
	 * 				CSV file
	 * @param proc 
	 * 				the {@link CSVReadProc} to use for CSV reading
	 */
	public void read(File file, CSVReadProc proc) {
		readAndClose(reader(file), proc);
	}
	
	/**
	 * Read CSV using the supplied {@link CSVReadProc}.
	 * 
	 * @param fileName 
	 * 				the file name
	 * @param proc 
	 * 				the {@link CSVReadProc} to use for CSV reading
	 */
	public void read(String fileName, CSVReadProc proc) {
		read(new File(fileName), proc);
	}
	
	/**
	 * Read CSV using the supplied {@link CSVReadProc}.
	 * 
	 * @param reader 
	 * 				the reader of an underlying CSV. The reader will not be closed.
	 * @param proc 
	 * 				the {@link CSVReadProc} to use for CSV reading
	 */	
	public void read(CSVReader reader, CSVReadProc proc) {
		reader.read(proc);
	}
	
	/**
	 * Read CSV using the supplied {@link CSVReadProc}.
	 * 
	 * @param is 
	 * 				the input stream of an underlying CSV. The input stream will be closed.
	 * @param proc 
	 * 				the {@link CSVReadProc} to use for CSV reading
	 */	
	public void readAndClose(InputStream is, CSVReadProc proc) {
		readAndClose(reader(is), proc);
	}
	
	/**
	 * Read CSV using the supplied {@link CSVReadProc}.
	 * 
	 * @param reader 
	 * 				the reader of an underlying CSV. The reader will be closed.
	 * @param proc 
	 * 				the {@link CSVReadProc} to use for CSV reading
	 */	
	public void readAndClose(Reader reader, CSVReadProc proc) {
		readAndClose(reader(reader), proc);
	}

	/**
	 * Read CSV using the supplied {@link CSVReadProc}.
	 * 
	 * @param reader 
	 * 				the reader of an underlying CSV. The reader will be closed.
	 * @param proc 
	 * 				the {@link CSVReadProc} to use for CSV reading
	 */	
	public void readAndClose(CSVReader reader, CSVReadProc proc) {
		try {
			read(reader, proc);
		} finally {
			try {
				reader.close();
			} catch (IOException e) {}
		}
	}

    /**
     * Constructs <code>CSV</code> using default separator, quote char, etc.
     */
	public static CSV create() {
		return new CSV();
	}
	
	/**
	 * Constructs <code>CSVBuilder</code> with specified separator
	 * 
	 * @param separator
	 * 			 the delimiter to use for separating entries
	 */
	public static Builder separator(char separator) {
		return new Builder().separator(separator);
	}

	/**
	 * Constructs <code>CSVBuilder</code> with specified quoteChar
	 * 
     * @param quoteChar
     *            the character to use for quoted elements
	 */
	public static Builder quote(char quoteChar) {
		return new Builder().quote(quoteChar);
	}
	
	/**
	 * Constructs <code>CSVBuilder</code> with no quote char
	 */
	public static Builder noQuote() {
		return new Builder().noQuote();
	}

	/**
	 * Constructs <code>CSVBuilder</code> with specified escapeChar
	 * 
     * @param escapeChar
     *            the character to use for escaping quotechars or escapechars
	 */
	public static Builder escape(char escapeChar) {
		return new Builder().escape(escapeChar);
	}
	
	/**
	 * Constructs <code>CSVBuilder</code> with no escape char
	 */
	public static Builder noEscape() {
		return new Builder().noEscape();
	}
	
	/**
	 * Constructs <code>CSVBuilder</code> with specified line terminator
	 * 
     * @param lineEnd
     * 			  the line feed terminator to use
	 */
	public static Builder lineEnd(String lineEnd) {
		return new Builder().lineEnd(lineEnd);
	}
	
	/**
	 * Constructs <code>CSVBuilder</code> with specified number of lines to skip for start reading
	 * 
     * @param skipLines
     *            the line number to skip for start reading
	 */
	public static Builder skipLines(int skipLines) {
		return new Builder().skipLines(skipLines);
	}

	/**
	 * Constructs <code>CSVBuilder</code>. Characters outside the quotes will be ignored.
	 */
	public static Builder strictQuotes() {
		return new Builder().strictQuotes();
	}

	/**
	 * Constructs <code>CSVBuilder</code>. Characters outside the quotes will not be ignored.
	 */
	public static Builder notStrictQuotes() {
		return new Builder().notStrictQuotes();
	}
	
	/**
	 * Constructs <code>CSVBuilder</code>. White space before a quote in a field will be ignored.
	 */
	public static Builder ignoreLeadingWhiteSpace() {
		return new Builder().ignoreLeadingWhiteSpace();
	}

	/**
	 * Constructs <code>CSVBuilder</code>. White space before a quote in a field will not be ignored.
	 */
	public static Builder notIgnoreLeadingWhiteSpace() {
		return new Builder().notIgnoreLeadingWhiteSpace();
	}
	
	/**
	 * Constructs <code>CSVBuilder</code> with specified charset
	 */
	public static Builder charset(Charset charset) {
		return new Builder().charset(charset);
	}
	
	/**
	 * Constructs <code>CSVBuilder</code> with specified charset
	 */
	public static Builder charset(String charsetName) {
		return new Builder().charset(charsetName);
	}

	
	  /**
	   * A builder for creating CSV formats. 
	   */
	public static class Builder {
	    private final CSV csv;
	    
		private Builder() {
			this.csv = new CSV();
		}

		private Builder(CSV csv) {
			this.csv = csv;
		}
		
	    /**
	     * Constructs <code>CSV</code>.
	     */
		public CSV create() {
			return csv;
		}
		
		/**
		 * Constructs <code>CSVBuilder</code> with specified separator
		 * 
		 * @param separator
		 * 			 the delimiter to use for separating entries
		 */
		public Builder separator(char separator) {
			return new Builder(new CSV(
					separator, 
					csv.quotechar, 
					csv.escapechar, 
					csv.lineEnd, 
					csv.skipLines,
					csv.strictQuotes,
					csv.ignoreLeadingWhiteSpace,
					csv.charset));
		}
		
		/**
		 * Constructs <code>CSVBuilder</code> with specified quoteChar
		 * 
	     * @param quoteChar
	     *            the character to use for quoted elements
		 */
		public Builder quote(char quoteChar) {
			return new Builder(new CSV(
					csv.separator, 
					quoteChar, 
					csv.escapechar, 
					csv.lineEnd, 
					csv.skipLines,
					csv.strictQuotes,
					csv.ignoreLeadingWhiteSpace,
					csv.charset));
		}
		
		/**
		 * Constructs <code>CSVBuilder</code> with specified escapeChar
		 * 
	     * @param escapeChar
	     *            the character to use for escaping quotechars or escapechars
		 */
		public Builder escape(char escapeChar) {
			return new Builder(new CSV(
					csv.separator, 
					csv.quotechar, 
					escapeChar, 
					csv.lineEnd, 
					csv.skipLines,
					csv.strictQuotes,
					csv.ignoreLeadingWhiteSpace,
					csv.charset));
		}
		
		/**
		 * Constructs <code>CSVBuilder</code> with specified line terminator
		 * 
	     * @param lineEnd
	     * 			  the line feed terminator to use
		 */
		public Builder lineEnd(String lineEnd) {
			return new Builder(new CSV(
					csv.separator, 
					csv.quotechar, 
					csv.escapechar, 
					lineEnd, 
					csv.skipLines,
					csv.strictQuotes,
					csv.ignoreLeadingWhiteSpace,
					csv.charset));
		}

		/**
		 * Constructs <code>CSVBuilder</code> with specified number of lines to skip for start reading
		 * 
	     * @param skipLines
	     *            the line number to skip for start reading
		 */
		public Builder skipLines(int skipLines) {
			return new Builder(new CSV(
					csv.separator, 
					csv.quotechar, 
					csv.escapechar, 
					csv.lineEnd, 
					skipLines,
					csv.strictQuotes,
					csv.ignoreLeadingWhiteSpace,
					csv.charset));
		}
		
		private Builder setStrictQuotes(boolean strictQuotes) {
			return new Builder(new CSV(
					csv.separator, 
					csv.quotechar, 
					csv.escapechar, 
					csv.lineEnd, 
					csv.skipLines,
					strictQuotes,
					csv.ignoreLeadingWhiteSpace,
					csv.charset));
		}

		private Builder setIgnoreLeadingWhiteSpace(boolean ignoreLeadingWhiteSpace) {
			return new Builder(new CSV(
					csv.separator, 
					csv.quotechar, 
					csv.escapechar, 
					csv.lineEnd, 
					csv.skipLines,
					csv.strictQuotes,
					ignoreLeadingWhiteSpace,					
					csv.charset));
		}

		/**
		 * Constructs <code>CSVBuilder</code> with specified charset
		 */
		public Builder charset(Charset charset) {
			return new Builder(new CSV(
					csv.separator, 
					csv.quotechar, 
					csv.escapechar, 
					csv.lineEnd, 
					csv.skipLines,
					csv.strictQuotes,
					csv.ignoreLeadingWhiteSpace,
					charset));
		}
		
		/**
		 * Constructs <code>CSVBuilder</code> with no quote char
		 */
		public Builder noQuote() {
			return quote(CSVWriter.NO_QUOTE_CHARACTER);
		}
		
		/**
		 * Constructs <code>CSVBuilder</code> with no escape char
		 */
		public Builder noEscape() {
			return escape(CSVWriter.NO_ESCAPE_CHARACTER);
		}
		
		/**
		 * Constructs <code>CSVBuilder</code>. Characters outside the quotes will be ignored.
		 */
		public Builder strictQuotes() {
			return setStrictQuotes(true);
		}
		
		/**
		 * Constructs <code>CSVBuilder</code>. Characters outside the quotes will not be ignored.
		 */
		public Builder notStrictQuotes() {
			return setStrictQuotes(false);
		}
		
		/**
		 * Constructs <code>CSVBuilder</code>. White space before a quote in a field will be ignored.
		 */
		public Builder ignoreLeadingWhiteSpace() {
			return setIgnoreLeadingWhiteSpace(true);
		}
		
		/**
		 * Constructs <code>CSVBuilder</code>. White space before a quote in a field will not be ignored.
		 */
		public Builder notIgnoreLeadingWhiteSpace() {
			return setIgnoreLeadingWhiteSpace(false);
		}

		/**
		 * Constructs <code>CSVBuilder</code> with specified charset
		 */
		public Builder charset(String charsetName) {
			return charset(Charset.forName(charsetName));
		}
	}
}
