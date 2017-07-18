/*-
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

package org.deeplearning4j.util;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;

import static org.deeplearning4j.berkeley.StringUtils.splitOnCharWithQuoting;


/**
 * String matrix
 * @author Adam Gibson
 *
 */
public class StringGrid extends ArrayList<List<String>> {


    private static final long serialVersionUID = 4702427632483221813L;
    private String sep;
    private int numColumns = -1;
    private static final Logger log = LoggerFactory.getLogger(StringGrid.class);
    public final static String NONE = "NONE";


    public StringGrid(StringGrid grid) {
        this.sep = grid.sep;
        this.numColumns = grid.numColumns;
        addAll(grid);
        fillOut();
    }

    public StringGrid(String sep, int numColumns) {
        this(sep, new ArrayList<String>());
        this.numColumns = numColumns;
        fillOut();
    }

    public int getNumColumns() {
        return numColumns;
    }

    private void fillOut() {
        for (List<String> list : this) {
            if (list.size() < numColumns) {
                int diff = numColumns - list.size();
                for (int i = 0; i < diff; i++) {
                    list.add(NONE);
                }
            }
        }
    }



    public static StringGrid fromFile(String file, String sep) throws IOException {
        List<String> read = FileUtils.readLines(new File(file));
        if (read.isEmpty())
            throw new IllegalStateException("Nothing to read; file is empty");

        return new StringGrid(sep, read);

    }

    public static StringGrid fromInput(InputStream from, String sep) throws IOException {
        List<String> read = IOUtils.readLines(from);
        if (read.isEmpty())
            throw new IllegalStateException("Nothing to read; file is empty");

        return new StringGrid(sep, read);

    }



    public StringGrid(String sep, Collection<String> data) {
        super();
        this.sep = sep;
        List<String> list = new ArrayList<>(data);
        for (int i = 0; i < list.size(); i++) {
            String line = list.get(i).trim();
            //text delimiter
            if (line.indexOf('\"') > 0) {
                Counter<Character> counter = new Counter<>();
                for (int j = 0; j < line.length(); j++) {
                    counter.incrementCount(line.charAt(j), 1.0f);
                }
                if (counter.getCount('"') > 1) {
                    String[] split = splitOnCharWithQuoting(line, sep.charAt(0), '"', '\\');
                    add(new ArrayList<>(Arrays.asList(split)));
                } else {
                    List<String> row = new ArrayList<>(
                                    Arrays.asList(splitOnCharWithQuoting(line, sep.charAt(0), '"', '\\')));
                    if (numColumns < 0)
                        numColumns = row.size();
                    else if (row.size() != numColumns)
                        log.warn("Row " + i + " had invalid number of columns  line was " + line);
                    add(row);
                }

            } else {
                List<String> row =
                                new ArrayList<>(Arrays.asList(splitOnCharWithQuoting(line, sep.charAt(0), '"', '\\')));
                if (numColumns < 0)
                    numColumns = row.size();
                else if (row.size() != numColumns) {
                    log.warn("Could not add " + line);
                }
                add(row);
            }

        }
        fillOut();
    }

    /**
     * Removes all rows with a column of NONE
     * @param column the column to remove by
     */
    public void removeRowsWithEmptyColumn(int column) {
        List<List<String>> remove = new ArrayList<>();
        for (List<String> list : this) {
            if (list.get(column).equals(NONE))
                remove.add(list);
        }
        removeAll(remove);
    }


    public void head(int num) {
        if (num >= size())
            num = size();
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < num; i++) {
            builder.append(get(i) + "\n");
        }

        log.info(builder.toString());
    }

    /**
     * Removes the specified columns from the grid
     * @param columns the columns to remove
     */
    public void removeColumns(Integer... columns) {
        if (columns.length < 1)
            throw new IllegalArgumentException("Columns must contain at least one column");
        List<Integer> removeOrder = Arrays.asList(columns);
        //put them in the right order for removing
        Collections.sort(removeOrder);
        for (List<String> list : this) {
            List<String> remove = new ArrayList<>();
            for (int i = 0; i < columns.length; i++) {
                remove.add(list.get(columns[i]));
            }
            list.removeAll(remove);
        }

    }

    /**
     * Removes all rows with a column of missingValue
     * @param column he column to remove by
     * @param missingValue the missingValue sentinel value
     */
    public void removeRowsWithEmptyColumn(int column, String missingValue) {
        List<List<String>> remove = new ArrayList<>();
        for (List<String> list : this) {
            if (list.get(column).equals(missingValue))
                remove.add(list);
        }
        removeAll(remove);
    }

    public List<List<String>> getRowsWithColumnValues(Collection<String> values, int column) {
        List<List<String>> ret = new ArrayList<>();
        for (List<String> val : this) {
            if (values.contains(val.get(column)))
                ret.add(val);
        }
        return ret;
    }

    public void sortColumnsByWordLikelihoodIncluded(final int column) {
        final Counter<String> counter = new Counter<>();
        List<String> col = getColumn(column);



        for (String s : col) {
            StringTokenizer tokenizer = new StringTokenizer(s);
            while (tokenizer.hasMoreTokens()) {
                counter.incrementCount(tokenizer.nextToken(), 1.0f);
            }
        }

        if (counter.totalCount() <= 0.0) {
            log.warn("Unable to calculate probability; nothing found");
            return;
        }

        //laplace smoothing
        counter.incrementAll(counter.keySet(), 1.0f);
        Set<String> remove = new HashSet<>();
        for (String key : counter.keySet())
            if (key.length() < 2 || key.matches("[a-z]+"))
                remove.add(key);
        for (String key : remove)
            counter.removeKey(key);

        counter.pruneKeysBelowThreshold(4.0f);


        final double totalCount = counter.totalCount();

        Collections.sort(this, new Comparator<List<String>>() {

            @Override
            public int compare(List<String> o1, List<String> o2) {
                double c1 = sumOverTokens(counter, o1.get(column), totalCount);
                double c2 = sumOverTokens(counter, o2.get(column), totalCount);
                return Double.compare(c1, c2);
            }

        });
    }

    /* Return the log sum of the column relative to the word frequencies (equivalent to the probability in log space */
    private double sumOverTokens(Counter<String> counter, String column, double totalCount) {
        StringTokenizer tokenizer = new StringTokenizer(column);
        double count = 0;
        while (tokenizer.hasMoreTokens())
            count += Math.log(counter.getCount(column) / totalCount);


        return count;
    }



    public StringCluster clusterColumn(int column) {
        return new StringCluster(getColumn(column));

    }

    public void dedupeByClusterAll() {
        for (int i = 0; i < size(); i++)
            dedupeByCluster(i);
    }

    /**
     * Deduplicate based on the column clustering signature
     * @param column
     */
    public void dedupeByCluster(int column) {
        StringCluster cluster = clusterColumn(column);
        System.out.println(cluster.get("family mcdonalds restaurant"));
        System.out.println(cluster.get("family mcdonalds restaurants"));
        List<Map<String, Integer>> list2 = cluster.getClusters();
        for (int i = 0; i < list2.size(); i++) {
            if (list2.get(i).size() > 1) {
                System.out.println(list2.get(i));
            }
        }
        FingerPrintKeyer keyer = new FingerPrintKeyer();
        Set<Integer> alreadyDeDupped = new HashSet<>();
        for (int i = 0; i < size(); i++) {
            String key = keyer.key(get(i).get(column));
            Map<String, Integer> map = cluster.get(key);
            if (map != null && map.size() > 1) {
                List<Integer> list = filterRowsByColumn(column, map.keySet());
                //deduplication to do
                if (list.size() > 1)
                    modifyRows(alreadyDeDupped, column, list, map);
            }


        }
    }

    /**
     * Cleans up the rows specified that haven't already been deduplified
     * @param alreadyDeDupped the already dedupped rows
     * @param column the column to homogenize
     * @param rows the rows to preProcess
     * @param cluster the cluster of values
     */
    private void modifyRows(Set<Integer> alreadyDeDupped, Integer column, List<Integer> rows,
                    Map<String, Integer> cluster) {
        String chosenKey = null;
        Integer max = null;
        for (Map.Entry<String, Integer> entry : cluster.entrySet()) {
            String key = entry.getKey();
            int value = entry.getValue();
            StringTokenizer val = new StringTokenizer(key);
            List<String> list = new ArrayList<>();
            boolean allLower = true;

            outer: while (val.hasMoreTokens()) {
                String token = val.nextToken();
                //weird capitalization
                if (token.length() >= 3 && token.matches("[A-Z]+")) {
                    continue outer;
                }
                list.add(token);
            }

            for (String s : list) {
                allLower = allLower && s.matches("[a-z]+");

            }
            if (allLower) {
                continue;
            }

            //not a proper name
            if (list.get(list.size() - 1).toLowerCase().equals("the")) {
                continue;
            }
            //first selection that's valid or count is higher
            if (max == null || (!allLower && value > max)) {
                max = value;
                chosenKey = key;
            }
        }

        //wtf is wrong with you people?
        if (chosenKey == null) {
            //getFromOrigin the max value of the cluster
            String max2 = maximalValue(cluster);
            StringTokenizer val = new StringTokenizer(max2);
            List<String> list = new ArrayList<>();
            while (val.hasMoreTokens()) {
                String token = val.nextToken();
                //weird capitalization
                if (token.length() >= 3 && token.matches("[A-Z]+")) {
                    token = token.charAt(0) + token.substring(1).toLowerCase();
                }
                list.add(token);
            }



            boolean allLower = true;
            for (String s : list)
                allLower = allLower && s.matches("[a-z]+");
            if (list.get(list.size() - 1).toLowerCase().equals("the")) {
                max2 = max2.replaceAll("^[Tt]he", "");

            }
            if (allLower)
                max2 = StringUtils.capitalize(max2);
            chosenKey = max2;



        }


        for (Integer i2 : rows) {
            //row already processed
            if (!alreadyDeDupped.contains(i2)) {
                disambiguateRow(i2, column, chosenKey);

            }
        }
    }

    private String maximalValue(Map<String, Integer> map) {
        Counter<String> counter = new Counter<>();
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            counter.incrementCount(entry.getKey(), entry.getValue());
        }
        return counter.argMax();
    }

    private void disambiguateRow(Integer row, Integer column, String chosenValue) {
        System.out.println("SETTING " + row + " column " + column + " to " + chosenValue);
        get(row).set(column, chosenValue);
    }



    public List<Integer> filterRowsByColumn(int column, Collection<String> values) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < size(); i++) {
            if (values.contains(get(i).get(column)))
                list.add(i);
        }
        return list;
    }



    public void sortBy(final int column) {
        Collections.sort(this, new Comparator<List<String>>() {

            @Override
            public int compare(List<String> o1, List<String> o2) {
                return o1.get(column).compareTo(o2.get(column));
            }

        });
    }

    public List<String> toLines() {
        List<String> lines = new ArrayList<>();
        for (List<String> list : this) {
            StringBuilder sb = new StringBuilder();
            for (String s : list) {
                sb.append(s.replaceAll(sep, " "));
                sb.append(sep);
            }

            lines.add(sb.toString().substring(0, sb.lastIndexOf(sep)));
        }
        return lines;
    }


    public void swap(int column1, int column2) {
        List<String> col1 = getColumn(column1);
        List<String> col2 = getColumn(column2);
        for (int i = 0; i < size(); i++) {
            get(i).set(column1, col2.get(i));
            get(i).set(column2, col1.get(i));
        }
    }

    public void merge(int column1, int column2) {
        checkInvalidColumn(column1);
        checkInvalidColumn(column2);

        if (column1 != column2)
            for (List<String> list : this) {
                StringBuilder sb = new StringBuilder();
                sb.append(list.get(column1));
                sb.append(list.get(column2));
                list.set(Math.min(column1, column2), sb.toString().replaceAll("\"", "").replace(sep, " "));
                list.remove(Math.max(column1, column2));
            }
        numColumns--;
    }


    public StringGrid getAllWithSimilarity(double threshold, int firstColumn, int secondColumn) {
        for (int column : new int[] {firstColumn, secondColumn})
            checkInvalidColumn(column);
        StringGrid grid = new StringGrid(sep, numColumns);
        for (List<String> list : this) {
            double sim = MathUtils.stringSimilarity(list.get(firstColumn), list.get(secondColumn));
            if (sim >= threshold)
                grid.addRow(list);
        }
        return grid;

    }

    public void writeLinesTo(String path) throws IOException {
        FileUtils.writeLines(new File(path), toLines());
    }


    public void fillDown(String value, int column) {
        checkInvalidColumn(column);
        for (List<String> list : this)
            list.set(column, value);
    }


    public StringGrid select(int column, String value) {
        StringGrid grid = new StringGrid(sep, numColumns);
        for (int i = 0; i < size(); i++) {
            List<String> row = get(i);
            if (row.get(column).equals(value)) {
                grid.addRow(row);
            }

        }
        return grid;
    }

    public void split(int column, String sepBy) {
        List<String> col = getColumn(column);
        int validate = -1;
        Set<String> remove = new HashSet<>();
        for (int i = 0; i < col.size(); i++) {
            String s = col.get(i);
            String[] split2 = StringUtils.splitOnCharWithQuoting(s, sepBy.charAt(0), '"', '\\');
            if (validate < 0)
                validate = split2.length;
            else if (validate != split2.length) {
                log.warn("Row " + get(i) + " will be invalid after split; removing");
                remove.add(s);
            }
        }

        for (String s : remove) {
            StringGrid grid = select(column, s);
            removeAll(grid);
        }
        Map<Integer, List<String>> replace = new HashMap<>();
        for (int i = 0; i < size(); i++) {
            List<String> list = get(i);
            List<String> newList = new ArrayList<>();
            String split = list.get(column);
            String[] split2 = StringUtils.splitOnCharWithQuoting(split, sepBy.charAt(0), '"', '\\');
            //add right next to where column was split
            for (int j = 0; j < list.size(); j++) {
                if (j == column)
                    for (String s : split2)
                        newList.add(s);

                else
                    newList.add(list.get(j));
            }
            replace.put(i, newList);


        }

        //prevent concurrent modification
        for (Map.Entry<Integer, List<String>> entry : replace.entrySet()) {
            set(entry.getKey(), entry.getValue());
        }
    }

    public void filterBySimilarity(double threshold, int firstColumn, int secondColumn) {
        for (int column : new int[] {firstColumn, secondColumn})
            checkInvalidColumn(column);
        List<List<String>> remove = new ArrayList<>();
        for (List<String> list : this) {
            double sim = MathUtils.stringSimilarity(list.get(firstColumn), list.get(secondColumn));
            if (sim < threshold)
                remove.add(list);
        }
        removeAll(remove);
    }

    public void prependToEach(String prepend, int toColumn) {
        for (List<String> row : this) {
            String currVal = row.get(toColumn);
            row.set(toColumn, prepend + currVal);
        }
    }

    public void appendToEach(String append, int toColumn) {
        for (List<String> row : this) {
            String currVal = row.get(toColumn);
            row.set(toColumn, currVal + append);
        }
    }

    public void addColumn(List<String> column) {
        if (column.size() != this.size())
            throw new IllegalArgumentException("Unable to add column; not enough rows");
        for (int i = 0; i < size(); i++) {
            get(i).add(column.get(i));
        }
    }

    /**
     * Combine the column based on a template and a number of template variable
     * columns. Note that this will also collapse the columns specified (removing them)
     *
     * @param templateColumn the column with the template ( uses printf style templating)
     * @param paramColumns the columns with template variables
     */
    public void combineColumns(int templateColumn, Integer[] paramColumns) {
        for (List<String> list : this) {
            List<String> format = new ArrayList<>();
            for (int j : paramColumns)
                format.add(list.get(j));


            list.set(templateColumn,
                            String.format(list.get(templateColumn), (Object[]) format.toArray(new String[] {})));
            //collapse columns
            list.removeAll(format);
        }
    }

    /**
     * Combine the column based on a template and a number of template variable
     * columns. Note that this will also collapse the columns specified (removing them)
     *
     * @param templateColumn the column with the template ( uses printf style templating)
     * @param paramColumns the columns with template variables
     */
    public void combineColumns(int templateColumn, int[] paramColumns) {
        for (List<String> list : this) {
            List<String> format = new ArrayList<>();
            for (int j : paramColumns)
                format.add(list.get(j));


            list.set(templateColumn,
                            String.format(list.get(templateColumn), (Object[]) format.toArray(new String[] {})));
            //collapse columns
            list.removeAll(format);
        }
    }

    public void addRow(List<String> row) {
        if (row.isEmpty()) {
            log.warn("Unable to add empty row");
        }

        else if (!isEmpty() && row.size() != get(0).size()) {
            log.warn("Unable to add row; not the same number of columns");
        } else
            add(row);
    }

    public Map<String, List<List<String>>> mapByPrimaryKey(int columnKey) {
        Map<String, List<List<String>>> map = new HashMap<>();
        for (List<String> line : this) {
            String val = line.get(columnKey);
            List<List<String>> get = map.get(val);
            if (get == null) {
                get = new ArrayList<>();
                map.put(val, get);
            }
            get.add(new ArrayList<>(Arrays.asList(sep)));

        }
        return map;
    }

    public List<String> getRow(int row) {
        checkInvalidRow(row);
        return new ArrayList<>(get(row));
    }

    public List<String> getColumn(int column) {
        checkInvalidColumn(column);
        List<String> ret = new ArrayList<>();
        for (List<String> list : this) {
            ret.add(list.get(column));
        }
        return ret;
    }

    private void checkInvalidRow(int row) {
        if (row < 0 || row >= size())
            throw new IllegalArgumentException("Row does not exist");
    }

    private void checkInvalidColumn(int column) {
        if (column < 0 || column >= numColumns)
            throw new IllegalArgumentException("Invalid column " + column);
    }


    public StringGrid getRowsWithDuplicateValuesInColumn(int column) {
        checkInvalidColumn(column);
        StringGrid grid = new StringGrid(sep, numColumns);
        List<String> columns = getColumn(column);
        Counter<String> counter = new Counter<>();
        for (String val : columns)
            counter.incrementCount(val, 1.0f);
        counter.pruneKeysBelowThreshold(2.0f);
        Set<String> keys = counter.keySet();
        for (List<String> row : this) {
            for (String key : keys)
                if (row.get(column).equals(key))
                    grid.addRow(row);

        }
        return grid;
    }

    public StringGrid getRowWithOnlyOneOccurrence(int column) {
        checkInvalidColumn(column);
        StringGrid grid = new StringGrid(sep, numColumns);
        List<String> columns = getColumn(column);
        Counter<String> counter = new Counter<>();
        for (String val : columns)
            counter.incrementCount(val, 1.0f);

        Set<String> keys = new HashSet<>(counter.keySet());
        for (String key : keys) {
            if (counter.getCount(key) > 1) {
                counter.removeKey(key);
            }
        }

        for (List<String> row : this) {
            for (String key : keys)
                if (row.get(column).equals(key))
                    grid.addRow(row);

        }
        return grid;
    }


    public StringGrid getUniqueRows() {
        StringGrid ret = new StringGrid(this);
        ret.stripDuplicateRows();
        return ret;
    }


    public void stripDuplicateRows() {
        Set<List<String>> set = new HashSet<>(this);
        clear();
        addAll(set);
    }



}
