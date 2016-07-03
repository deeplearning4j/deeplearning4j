/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.berkeley;


import java.io.*;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.net.URLConnection;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * StringUtils is a class for random String things.
 *
 * @author Dan Klein
 * @author Christopher Manning
 * @author Tim Grow (grow@stanford.edu)
 * @author Chris Cox
 * @version 2003/02/03
 */
public class StringUtils {

	/**
	 * Don't let anyone instantiate this class.
	 */
	private StringUtils() {
	}

	/**
	 * Say whether this regular expression can be found inside
	 * this String.  This method provides one of the two "missing"
	 * convenience methods for regular expressions in the String class
	 * in JDK1.4.  This is the one you'll want to use all the time if
	 * you're used to Perl.  What were they smoking?
	 *
	 * @param str   String to search for match in
	 * @param regex String to compile as the regular expression
	 * @return Whether the regex can be found in str
	 */
	public static boolean find(String str, String regex) {
		return Pattern.compile(regex).matcher(str).find();
	}

	/**
	 * Say whether this regular expression can be found at the beginning of
	 * this String.  This method provides one of the two "missing"
	 * convenience methods for regular expressions in the String class
	 * in JDK1.4.
	 *
	 * @param str   String to search for match at start of
	 * @param regex String to compile as the regular expression
	 * @return Whether the regex can be found at the start of str
	 */
	public static boolean lookingAt(String str, String regex) {
		return Pattern.compile(regex).matcher(str).lookingAt();
	}

	/**
	 * Say whether this regular expression matches
	 * this String.  This method is the same as the String.matches() method,
	 * and is included just to give a call that is parallel to the other
	 * static regex methods in this class.
	 *
	 * @param str   String to search for match at start of
	 * @param regex String to compile as the regular expression
	 * @return Whether the regex matches the whole of this str
	 */
	public static boolean matches(String str, String regex) {
		return Pattern.compile(regex).matcher(str).matches();
	}

	private static final int SLURPBUFFSIZE = 16000;

	/**
	 * Returns all the text in the given File.
	 */
	public static String slurpFile(File file) throws IOException {
		Reader r = new FileReader(file);
		return slurpReader(r);
	}

	public static String slurpGBFileNoExceptions(String filename) {
		return slurpFileNoExceptions(filename, "GB18030");
	}

	/**
	 * Returns all the text in the given file with the given encoding.
	 */
	public static String slurpFile(String filename, String encoding) throws IOException {
		Reader r = new InputStreamReader(new FileInputStream(filename), encoding);
		return slurpReader(r);
	}

	/**
	 * Returns all the text in the given file with the given encoding.
	 * If the file cannot be read (non-existent, etc.),
	 * then and only then the method returns <code>null</code>.
	 */
	public static String slurpFileNoExceptions(String filename, String encoding) {
		try {
			return slurpFile(filename, encoding);
		} catch (Exception e) {
			throw new RuntimeException();
		}
	}

	public static String slurpGBFile(String filename) throws IOException {
		return slurpFile(filename, "GB18030");
	}

	/**
	 * Returns all the text from the given Reader.
	 *
	 * @return The text in the file.
	 */
	public static String slurpReader(Reader reader) {
		BufferedReader r = new BufferedReader(reader);
		StringBuilder buff = new StringBuilder();
		try {
			char[] chars = new char[SLURPBUFFSIZE];
			while (true) {
				int amountRead = r.read(chars, 0, SLURPBUFFSIZE);
				if (amountRead < 0) {
					break;
				}
				buff.append(chars, 0, amountRead);
			}
			r.close();
		} catch (Exception e) {
			throw new RuntimeException();
		}
		return buff.toString();
	}

	/**
	 * Returns all the text in the given file
	 *
	 * @return The text in the file.
	 */
	public static String slurpFile(String filename) throws IOException {
		return slurpReader(new FileReader(filename));
	}

	/**
	 * Returns all the text in the given File.
	 *
	 * @return The text in the file.  May be an empty string if the file
	 *         is empty.  If the file cannot be read (non-existent, etc.),
	 *         then and only then the method returns <code>null</code>.
	 */
	public static String slurpFileNoExceptions(File file) {
		try {
			return slurpReader(new FileReader(file));
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * Returns all the text in the given File.
	 *
	 * @return The text in the file.  May be an empty string if the file
	 *         is empty.  If the file cannot be read (non-existent, etc.),
	 *         then and only then the method returns <code>null</code>.
	 */
	public static String slurpFileNoExceptions(String filename) {
		try {
			return slurpFile(filename);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * Returns all the text at the given URL.
	 */
	public static String slurpGBURL(URL u) throws IOException {
		return slurpURL(u, "GB18030");
	}

	/**
	 * Returns all the text at the given URL.
	 */
	public static String slurpGBURLNoExceptions(URL u) {
		try {
			return slurpGBURL(u);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * Returns all the text at the given URL.
	 */
	public static String slurpURLNoExceptions(URL u, String encoding) {
		try {
			return slurpURL(u, encoding);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * Returns all the text at the given URL.
	 */
	public static String slurpURL(URL u, String encoding) throws IOException {
		String lineSeparator = System.getProperty("line.separator");
		URLConnection uc = u.openConnection();
		uc.setReadTimeout(30000);
		InputStream is;
		try {
			is = uc.getInputStream();
		} catch (SocketTimeoutException e) {
			//e.printStackTrace();
			System.err.println("Time out. Return empty string");
			return "";
		}
		BufferedReader br = new BufferedReader(new InputStreamReader(is, encoding));
		String temp;
		StringBuilder buff = new StringBuilder(16000); // make biggish
		while ((temp = br.readLine()) != null) {
			buff.append(temp);
			buff.append(lineSeparator);
		}
		br.close();
		return buff.toString();
	}

	/**
	 * Returns all the text at the given URL.
	 */
	public static String slurpURL(URL u) throws IOException {
		String lineSeparator = System.getProperty("line.separator");
		URLConnection uc = u.openConnection();
		InputStream is = uc.getInputStream();
		BufferedReader br = new BufferedReader(new InputStreamReader(is));
		String temp;
		StringBuilder buff = new StringBuilder(16000); // make biggish
		while ((temp = br.readLine()) != null) {
			buff.append(temp);
			buff.append(lineSeparator);
		}
		br.close();
		return buff.toString();
	}

	/**
	 * Returns all the text at the given URL.
	 */
	public static String slurpURLNoExceptions(URL u) {
		try {
			return slurpURL(u);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * Returns all the text at the given URL.
	 */
	public static String slurpURL(String path) throws Exception {
		return slurpURL(new URL(path));
	}

	/**
	 * Returns all the text at the given URL. If the file cannot be read (non-existent, etc.),
	 * then and only then the method returns <code>null</code>.
	 */
	public static String slurpURLNoExceptions(String path) {
		try {
			return slurpURL(path);
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * Joins each elem in the Collection with the given glue. For example, given a
	 * list
	 * of Integers, you can createComplex a comma-separated list by calling
	 * <tt>join(numbers, ", ")</tt>.
	 */
	public static String join(Iterable l, String glue) {
		StringBuilder sb = new StringBuilder();
		boolean first = true;
		for (Object o : l) {
			if (!first) {
				sb.append(glue);
			}
			sb.append(o.toString());
			first = false;
		}
		return sb.toString();
	}

	/**
	 * Joins each elem in the List with the given glue. For example, given a
	 * list
	 * of Integers, you can createComplex a comma-separated list by calling
	 * <tt>join(numbers, ", ")</tt>.
	 */
	public static String join(List<?> l, String glue) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < l.size(); i++) {
			if (i > 0) {
				sb.append(glue);
			}
			Object x = l.get(i);
			sb.append(x.toString());
		}
		return sb.toString();
	}

	/**
	 * Joins each elem in the array with the given glue. For example, given a list
	 * of ints, you can createComplex a comma-separated list by calling
	 * <tt>join(numbers, ", ")</tt>.
	 */
	public static String join(Object[] elements, String glue) {
		return (join(Arrays.asList(elements), glue));
	}

	/**
	 * Joins elems with a space.
	 */
	public static String join(List l) {
		return join(l, " ");
	}

	/**
	 * Joins elems with a space.
	 */
	public static String join(Object[] elements) {
		return (join(elements, " "));
	}

	/**
	 * Splits on whitespace (\\s+).
	 */
	public static List split(String s) {
		return (split(s, "\\s+"));
	}

	/**
	 * Splits the given string using the given regex as delimiters.
	 * This method is the same as the String.split() method (except it throws
	 * the results in a List),
	 * and is included just to give a call that is parallel to the other
	 * static regex methods in this class.
	 *
	 * @param str   String to split up
	 * @param regex String to compile as the regular expression
	 * @return List of Strings resulting from splitting on the regex
	 */
	public static List split(String str, String regex) {
		return (Arrays.asList(str.split(regex)));
	}

	/**
	 * Return a String of length a minimum of totalChars characters by
	 * padding the input String str with spaces.  If str is already longer
	 * than totalChars, it is returned unchanged.
	 */
	public static String pad(String str, int totalChars) {
		if (str == null)
			str = "null";
		int slen = str.length();
		StringBuilder sb = new StringBuilder(str);
		for (int i = 0; i < totalChars - slen; i++) {
			sb.append(" ");
		}
		return sb.toString();
	}

	/**
	 * Pads the toString value of the given Object.
	 */
	public static String pad(Object obj, int totalChars) {
		return pad(obj.toString(), totalChars);
	}

	/**
	 * Pad or trim so as to produce a string of exactly a certain length.
	 *
	 * @param str The String to be padded or truncated
	 * @param num The desired length
	 */
	public static String padOrTrim(String str, int num) {
		if (str == null)
			str = "null";
		int leng = str.length();
		if (leng < num) {
			StringBuilder sb = new StringBuilder(str);
			for (int i = 0; i < num - leng; i++) {
				sb.append(" ");
			}
			return sb.toString();
		} else if (leng > num) {
			return str.substring(0, num);
		} else {
			return str;
		}
	}

	/**
	 * Pad or trim the toString value of the given Object.
	 */
	public static String padOrTrim(Object obj, int totalChars) {
		return padOrTrim(obj.toString(), totalChars);
	}

	/**
	 * Pads the given String to the left with spaces to ensure that it's
	 * at least totalChars long.
	 */
	public static String padLeft(String str, int totalChars) {
		if (str == null)
			str = "null";
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < totalChars - str.length(); i++) {
			sb.append(" ");
		}
		sb.append(str);
		return sb.toString();
	}

	public static String padLeft(Object obj, int totalChars) {
		return padLeft(obj.toString(), totalChars);
	}

	public static String padLeft(int i, int totalChars) {
		return padLeft(new Integer(i), totalChars);
	}

	public static String padLeft(double d, int totalChars) {
		return padLeft(new Double(d), totalChars);
	}

	/**
	 * Returns s if it's at most maxWidth chars, otherwise chops right side to fit.
	 */
	public static String trim(String s, int maxWidth) {
		if (s.length() <= maxWidth) {
			return (s);
		}
		return (s.substring(0, maxWidth));
	}

	public static String trim(Object obj, int maxWidth) {
		return trim(obj.toString(), maxWidth);
	}

	/**
	 * Returns a "clean" version of the given filename in which spaces have
	 * been converted to dashes and all non-alphaneumeric chars are underscores.
	 */
	public static String fileNameClean(String s) {
		char[] chars = s.toCharArray();
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < chars.length; i++) {
			char c = chars[i];
			if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
					|| (c == '_')) {
				sb.append(c);
			} else {
				if (c == ' ' || c == '-') {
					sb.append('_');
				} else {
					sb.append("x" + (int) c + "x");
				}
			}
		}
		return sb.toString();
	}

	/**
	 * Returns the index of the <i>n</i>th occurrence of ch in s, or -1
	 * if there are less than n occurrences of ch.
	 */
	public static int nthIndex(String s, char ch, int n) {
		int index = 0;
		for (int i = 0; i < n; i++) {
			// if we're already at the end of the string,
			// and we need to find another ch, return -1
			if (index == s.length() - 1) {
				return -1;
			}
			index = s.indexOf(ch, index + 1);
			if (index == -1) {
				return (-1);
			}
		}
		return index;
	}

	/**
	 * This returns a string from decimal digit smallestDigit to decimal digit
	 * biggest digit. Smallest digit is labeled 1, and the limits are
	 * inclusive.
	 */
	public static String truncate(int n, int smallestDigit, int biggestDigit) {
		int numDigits = biggestDigit - smallestDigit + 1;
		char[] result = new char[numDigits];
		for (int j = 1; j < smallestDigit; j++) {
			n = n / 10;
		}
		for (int j = numDigits - 1; j >= 0; j--) {
			result[j] = Character.forDigit(n % 10, 10);
			n = n / 10;
		}
		return new String(result);
	}

	/**
	 * Parses command line arguments into a Map. Arguments of the form
	 * <p/>
	 * -flag1 arg1a arg1b ... arg1m -flag2 -flag3 arg3a ... arg3n
	 * <p/>
	 * will be parsed so that the flag is a key in the Map (including
	 * the hyphen) and its value will be a {@link String[] } containing
	 * the optional arguments (if present).  The non-flag values not
	 * captured as flag arguments are collected into a String[] array
	 * and returned as the value of <code>null</code> in the Map.  In
	 * this invocation, flags cannot take arguments, so all the {@link
	 * String} array values other than the value for <code>null</code>
	 * will be zero-length.
	 *
	 * @param args
	 * @return a {@link Map} of flag names to flag argument {@link
	 *         String[]} arrays.
	 */
	public static Map<String, String[]> argsToMap(String[] args) {
		return argsToMap(args, new HashMap<String, Integer>());
	}

	/**
	 * Parses command line arguments into a Map. Arguments of the form
	 * <p/>
	 * -flag1 arg1a arg1b ... arg1m -flag2 -flag3 arg3a ... arg3n
	 * <p/>
	 * will be parsed so that the flag is a key in the Map (including
	 * the hyphen) and its value will be a {@link String[] } containing
	 * the optional arguments (if present).  The non-flag values not
	 * captured as flag arguments are collected into a String[] array
	 * and returned as the value of <code>null</code> in the Map.  In
	 * this invocation, the maximum number of arguments for each flag
	 * can be specified as an {@link Integer} value of the appropriate
	 * flag key in the <code>flagsToNumArgs</code> {@link Map}
	 * argument. (By default, flags cannot take arguments.)
	 * <p/>
	 * Example of usage:
	 * <p/>
	 * <code>
	 * Map flagsToNumArgs = new HashMap();
	 * flagsToNumArgs.put("-x",new Integer(2));
	 * flagsToNumArgs.put("-d",new Integer(1));
	 * Map result = argsToMap(args,flagsToNumArgs);
	 * </code>
	 *
	 * @param args           the argument array to be parsed
	 * @param flagsToNumArgs a {@link Map} of flag names to {@link
	 *                       Integer} values specifying the maximum number of allowed
	 *                       arguments for that flag (default 0).
	 * @return a {@link Map} of flag names to flag argument {@link
	 *         String[]} arrays.
	 */
	public static Map<String, String[]> argsToMap(String[] args,
			Map<String, Integer> flagsToNumArgs) {
		Map<String, String[]> result = new HashMap<>();
		List<String> remainingArgs = new ArrayList<>();
		String key;
		for (int i = 0; i < args.length; i++) {
			key = args[i];
			if (key.charAt(0) == '-') { // found a flag
				Integer maxFlagArgs = flagsToNumArgs.get(key);
				int max = maxFlagArgs == null ? 0 : maxFlagArgs.intValue();
				List<String> flagArgs = new ArrayList<>();
				for (int j = 0; j < max && i + 1 < args.length && args[i + 1].charAt(0) != '-'; i++, j++) {
					flagArgs.add(args[i + 1]);
				}
				if (result.containsKey(key)) { // append the second specification into the args.
					String[] newFlagArg = new String[result.get(key).length
							+ flagsToNumArgs.get(key)];
					int oldNumArgs = result.get(key).length;
					System.arraycopy(result.get(key), 0, newFlagArg, 0, oldNumArgs);
					for (int j = 0; j < flagArgs.size(); j++) {
						newFlagArg[j + oldNumArgs] = flagArgs.get(j);
					}
				} else
					result.put(key, flagArgs.toArray(new String[] {}));
			} else {
				remainingArgs.add(args[i]);
			}
		}
		result.put(null, remainingArgs.toArray(new String[] {}));
		return result;
	}

	private static final String PROP = "prop";

	public static Properties argsToProperties(String[] args) {
		return argsToProperties(args, new HashMap());
	}

	/**
	 * Analagous to {@link #argsToMap}.  However, there are several key differences between this method and {@link #argsToMap}:
	 * <ul>
	 * <li> Hyphens are stripped from flag names </li>
	 * <li> Since Properties objects are String to String mappings, the default number of arguments to a flag is
	 * assumed to be 1 and not 0. </li>
	 * <li> Furthermore, the list of arguments not bound to a flag is mapped to the "" property, not null </li>
	 * <li> The special flag "-prop" will load the property file specified by it's argument. </li>
	 * <li> The value for flags without arguments is applyTransformToDestination to "true" </li>
	 */
	public static Properties argsToProperties(String[] args, Map flagsToNumArgs) {
		Properties result = new Properties();
		List<String> remainingArgs = new ArrayList<>();
		String key;
		for (int i = 0; i < args.length; i++) {
			key = args[i];
			if (key.charAt(0) == '-') { // found a flag
				key = key.substring(1); // strip off the hyphen

				Integer maxFlagArgs = (Integer) flagsToNumArgs.get(key);
				int max = maxFlagArgs == null ? 1 : maxFlagArgs.intValue();
				List<String> flagArgs = new ArrayList<>();
				for (int j = 0; j < max && i + 1 < args.length && args[i + 1].charAt(0) != '-'; i++, j++) {
					flagArgs.add(args[i + 1]);
				}
				if (flagArgs.isEmpty()) {
					result.setProperty(key, "true");
				} else {
					result.setProperty(key, join(flagArgs, " "));
					if (key.equalsIgnoreCase(PROP)) {
						try {
							result.load(new BufferedInputStream(new FileInputStream(result
									.getProperty(PROP))));
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
				}
			} else {
				remainingArgs.add(args[i]);
			}
		}
		result.setProperty("", join(remainingArgs, " "));
		return result;
	}

	/**
	 * This method converts a comma-separated String (with whitespace
	 * optionally allowed after the comma) representing properties
	 * to a Properties object.  Each property is "property=value".  The value
	 * for properties without an explicitly given value is applyTransformToDestination to "true".
	 */
	public static Properties stringToProperties(String str) {
		Properties result = new Properties();
		String[] props = str.trim().split(",\\s*");
		for (int i = 0; i < props.length; i++) {
			String term = props[i];
			int divLoc = term.indexOf('=');
			String key;
			String value;
			if (divLoc >= 0) {
				key = term.substring(0, divLoc);
				value = term.substring(divLoc + 1);
			} else {
				key = term;
				value = "true";
			}
			result.setProperty(key, value);
		}
		return result;
	}

	/**
	 * Prints to a file.  If the file already exists, appends if
	 * <code>append=true</code>, and overwrites if <code>append=false</code>
	 */
	public static void printToFile(File file, String message, boolean append) {
		FileWriter fw = null;
		PrintWriter pw = null;
		try {
			fw = new FileWriter(file, append);
			pw = new PrintWriter(fw);
			pw.print(message);
		} catch (Exception e) {
			System.out.println("Exception: in printToFile " + file.getAbsolutePath() + " "
					+ message);
			e.printStackTrace();
		} finally {
			if (pw != null) {
				pw.close();
			}
		}
	}

	/**
	 * Prints to a file.  If the file does not exist, rewrites the file;
	 * does not append.
	 */
	public static void printToFile(File file, String message) {
		printToFile(file, message, false);
	}

	/**
	 * Prints to a file.  If the file already exists, appends if
	 * <code>append=true</code>, and overwrites if <code>append=false</code>
	 */
	public static void printToFile(String filename, String message, boolean append) {
		printToFile(new File(filename), message, append);
	}

	/**
	 * Prints to a file.  If the file does not exist, rewrites the file;
	 * does not append.
	 */
	public static void printToFile(String filename, String message) {
		printToFile(new File(filename), message, false);
	}

	/**
	 * A simpler form of command line argument parsing.
	 * Dan thinks this is highly superior to the overly complexified code that
	 * comes before it.
	 * Parses command line arguments into a Map. Arguments of the form
	 * -flag1 arg1 -flag2 -flag3 arg3
	 * will be parsed so that the flag is a key in the Map (including the hyphen)
	 * and the
	 * optional argument will be its value (if present).
	 *
	 * @param args
	 * @return A Map from keys to possible values (String or null)
	 */
	public static Map parseCommandLineArguments(String[] args) {
		Map<String, String> result = new HashMap<>();
		String key, value;
		for (int i = 0; i < args.length; i++) {
			key = args[i];
			if (key.charAt(0) == '-') {
				if (i + 1 < args.length) {
					value = args[i + 1];
					if (value.charAt(0) != '-') {
						result.put(key, value);
						i++;
					} else {
						result.put(key, null);
					}
				} else {
					result.put(key, null);
				}
			}
		}
		return result;
	}

	public static String stripNonAlphaNumerics(String orig) {
		StringBuilder sb = new StringBuilder();
		char c;
		for (int i = 0; i < orig.length(); i++) {
			c = orig.charAt(i);
			if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
				sb.append(c);
			}
		}
		return sb.toString();
	}

	public static void printStringOneCharPerLine(String s) {
		for (int i = 0; i < s.length(); i++) {
			int c = s.charAt(i);
			System.out.println(c + " \'" + (char) c + "\' ");
		}
	}

	public static String escapeString(String s, char[] charsToEscape, char escapeChar) {
		StringBuilder result = new StringBuilder();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (c == escapeChar) {
				result.append(escapeChar);
			} else {
				for (int j = 0; j < charsToEscape.length; j++) {
					if (c == charsToEscape[j]) {
						result.append(escapeChar);
						break;
					}
				}
			}
			result.append(c);
		}
		return result.toString();
	}

	/**
	 * This function splits the String s into multiple Strings using the
	 * splitChar.  However, it provides an quoting facility: it is possible to
	 * quote strings with the quoteChar.
	 * If the quoteChar occurs within the quotedExpression, it must be prefaced
	 * by the escapeChar
	 *
	 * @param s         The String to split
	 * @param splitChar
	 * @param quoteChar
	 * @return An array of Strings that s is split into
	 */
	public static String[] splitOnCharWithQuoting(String s, char splitChar, char quoteChar,
			char escapeChar) {
		List<String> result = new ArrayList<>();
		int i = 0;
		int length = s.length();
		StringBuilder b = new StringBuilder();
		while (i < length) {
			char curr = s.charAt(i);
			if (curr == splitChar) {
				// add last buffer
				if (b.length() > 0) {
					result.add(b.toString());
					b = new StringBuilder();
				}
				i++;
			} else if (curr == quoteChar) {
				// find next instance of quoteChar
				i++;
				while (i < length) {
					curr = s.charAt(i);
					if (curr == escapeChar) {
						b.append(s.charAt(i + 1));
						i += 2;
					} else if (curr == quoteChar) {
						i++;
						break; // break this loop
					} else {
						b.append(s.charAt(i));
						i++;
					}
				}
			} else {
				b.append(curr);
				i++;
			}
		}
		if (b.length() > 0) {
			result.add(b.toString());
		}
		return result.toArray(new String[0]);
	}

	/**
	 * Computes the longest common substring of s and t.
	 * The longest common substring of a and b is the longest run of
	 * characters that appear in order inside both a and b. Both a and b
	 * may have other extraneous characters along the way. This is like
	 * edit distance but with no substitution and a higher number means
	 * more similar. For example, the LCS of "abcD" and "aXbc" is 3 (abc).
	 */
	public static int longestCommonSubstring(String s, String t) {
		int d[][]; // matrix
		int n; // length of s
		int m; // length of t
		int i; // iterates through s
		int j; // iterates through t
		char s_i; // ith character of s
		char t_j; // jth character of t
		// Step 1
		n = s.length();
		m = t.length();
		if (n == 0) {
			return 0;
		}
		if (m == 0) {
			return 0;
		}
		d = new int[n + 1][m + 1];
		// Step 2
		for (i = 0; i <= n; i++) {
			d[i][0] = 0;
		}
		for (j = 0; j <= m; j++) {
			d[0][j] = 0;
		}
		// Step 3
		for (i = 1; i <= n; i++) {
			s_i = s.charAt(i - 1);
			// Step 4
			for (j = 1; j <= m; j++) {
				t_j = t.charAt(j - 1);
				// Step 5
				// js: if the chars match, you can getFromOrigin an extra point
				// otherwise you have to skip an insertion or deletion (no subs)
				if (s_i == t_j) {
					d[i][j] = SloppyMath.max(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1] + 1);
				} else {
					d[i][j] = Math.max(d[i - 1][j], d[i][j - 1]);
				}
			}
		}
		if (false) {
			// num chars needed to display longest num
			int numChars = (int) Math.ceil(Math.log(d[n][m]) / Math.log(10));
			for (i = 0; i < numChars + 3; i++) {
				System.err.print(' ');
			}
			for (j = 0; j < m; j++) {
				System.err.print("" + t.charAt(j) + " ");
			}
			System.err.println();
			for (i = 0; i <= n; i++) {
				System.err.print((i == 0 ? ' ' : s.charAt(i - 1)) + " ");
				for (j = 0; j <= m; j++) {
					System.err.print("" + d[i][j] + " ");
				}
				System.err.println();
			}
		}
		// Step 7
		return d[n][m];
	}

	/**
	 * Computes the Levenshtein (edit) distance of the two given Strings.
	 */
	public static int editDistance(String s, String t) {
		int d[][]; // matrix
		int n; // length of s
		int m; // length of t
		int i; // iterates through s
		int j; // iterates through t
		char s_i; // ith character of s
		char t_j; // jth character of t
		int cost; // cost
		// Step 1
		n = s.length();
		m = t.length();
		if (n == 0) {
			return m;
		}
		if (m == 0) {
			return n;
		}
		d = new int[n + 1][m + 1];
		// Step 2
		for (i = 0; i <= n; i++) {
			d[i][0] = i;
		}
		for (j = 0; j <= m; j++) {
			d[0][j] = j;
		}
		// Step 3
		for (i = 1; i <= n; i++) {
			s_i = s.charAt(i - 1);
			// Step 4
			for (j = 1; j <= m; j++) {
				t_j = t.charAt(j - 1);
				// Step 5
				if (s_i == t_j) {
					cost = 0;
				} else {
					cost = 1;
				}
				// Step 6
				d[i][j] = SloppyMath
						.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
			}
		}

		// Step 7
		return d[n][m];
	}

	/**
	 * Computes the WordNet 2.0 POS tag corresponding to the PTB POS tag s.
	 *
	 * @param s a Penn TreeBank POS tag.
	 */
	public static String pennPOSToWordnetPOS(String s) {
		if (s.matches("NN|NNP|NNS|NNPS")) {
			return "noun";
		}
		if (s.matches("VB|VBD|VBG|VBN|VBZ|VBP|MD")) {
			return "verb";
		}
		if (s.matches("JJ|JJR|JJS|CD")) {
			return "adjective";
		}
		if (s.matches("RB|RBR|RBS|RP|WRB")) {
			return "adverb";
		}
		return null;
	}

	/**
	 * Uppercases the first character of a string.
	 *
	 * @param s a string to capitalize
	 * @return a capitalized version of the string
	 */
	public static String capitalize(String s) {
		if (s.charAt(0) >= 'a') {
			return (char) (s.charAt(0) + ('A' - 'a')) + s.substring(1);
		} else {
			return s;
		}
	}

  public static List<Matcher> allMatches(String str, String regex) {
    Pattern p = Pattern.compile(regex);
    List<Matcher> matches = new ArrayList<>();
    while (true) {
      Matcher m = p.matcher(str);
      if (!m.find()) break;
      matches.add(m);
      str = str.substring(m.end());
    }
    return matches;
  }

	public static void main(String[] args) throws IOException {

		String[] s = { "there once was a man", "this one is a manic", "hey there",
				"there once was a mane", "once in a manger.", "where is one match?" };
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				System.out.println("s1: " + s[i]);
				System.out.println("s2: " + s[j]);
				System.out.println("edit distance: " + editDistance(s[i], s[j]));
				System.out.println("LCS:           " + longestCommonSubstring(s[i], s[j]));
				System.out.println();
			}
		}
	}

}
