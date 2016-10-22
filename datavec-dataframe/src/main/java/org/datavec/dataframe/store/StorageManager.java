package org.datavec.dataframe.store;

import org.datavec.dataframe.api.BooleanColumn;
import org.datavec.dataframe.api.CategoryColumn;
import org.datavec.dataframe.api.DateColumn;
import org.datavec.dataframe.api.DateTimeColumn;
import org.datavec.dataframe.api.FloatColumn;
import org.datavec.dataframe.api.IntColumn;
import org.datavec.dataframe.api.LongColumn;
import org.datavec.dataframe.api.ShortColumn;
import org.datavec.dataframe.api.Table;
import org.datavec.dataframe.api.TimeColumn;
import org.datavec.dataframe.columns.Column;
import org.datavec.dataframe.table.Relation;
import org.iq80.snappy.SnappyFramedInputStream;
import org.iq80.snappy.SnappyFramedOutputStream;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystem;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.concurrent.CompletionService;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.regex.Pattern;

/**
 * A controller for reading and writing data in Tablesaw's own compressed, column-oriented file format
 */
public class StorageManager {

  private static final int FLUSH_AFTER_ITERATIONS = 10_000;

  private static final String FILE_EXTENSION = "saw";
  private static final Pattern WHITE_SPACE_PATTERN = Pattern.compile("\\s+");
  private static final Pattern SEPARATOR_PATTERN = Pattern.compile(separator());

  private static final int READER_POOL_SIZE = 4;

  static String separator() {
    FileSystem fileSystem = FileSystems.getDefault();
    return fileSystem.getSeparator();
  }

  /**
   * Reads a tablesaw table into memory
   *
   * @param path The location of the table. It is interpreted as relative to the working directory if not fully
   *             specified. The path will typically end in ".saw", as in "mytables/nasdaq-2015.saw"
   * @throws IOException if the file cannot be read
   */
  public static Table readTable(String path) throws IOException {

    ExecutorService executorService = Executors.newFixedThreadPool(READER_POOL_SIZE);
    CompletionService readerCompletionService = new ExecutorCompletionService<>(executorService);

    TableMetadata tableMetadata = readTableMetadata(path + separator() + "Metadata.json");
    List<ColumnMetadata> columnMetadata = tableMetadata.getColumnMetadataList();
    Table table = Table.create(tableMetadata);

    // NB: We do some extra work with the hash map to ensure that the columns are added to the table in original order
    // TODO(lwhite): Not using CPU efficiently. Need to prevent waiting for other threads until all columns are read
    // TODO - continued : Problem seems to be mostly with category columns rebuilding the encoding dictionary
    ConcurrentLinkedQueue<Column> columnList = new ConcurrentLinkedQueue<>();
    Map<String, Column> columns = new HashMap<>();
    try {
      for (ColumnMetadata column : columnMetadata) {
        readerCompletionService.submit(() -> {
          columnList.add(readColumn(path + separator() + column.getId(), column));
          return null;
        });
      }
      for (int i = 0; i < columnMetadata.size(); i++) {
        Future future = readerCompletionService.take();
        future.get();
      }
      for (Column c : columnList) {
        columns.put(c.id(), c);
      }

      for (ColumnMetadata metadata : columnMetadata) {
        String id = metadata.getId();
        table.addColumn(columns.get(id));
      }

    } catch (InterruptedException | ExecutionException e) {
      throw new RuntimeException(e);
    }
    executorService.shutdown();
    return table;
  }

  private static Column readColumn(String fileName, ColumnMetadata columnMetadata)
      throws IOException {

    switch (columnMetadata.getType()) {
      case FLOAT:
        return readFloatColumn(fileName, columnMetadata);
      case INTEGER:
        return readIntColumn(fileName, columnMetadata);
      case BOOLEAN:
        return readBooleanColumn(fileName, columnMetadata);
      case LOCAL_DATE:
        return readLocalDateColumn(fileName, columnMetadata);
      case LOCAL_TIME:
        return readLocalTimeColumn(fileName, columnMetadata);
      case LOCAL_DATE_TIME:
        return readLocalDateTimeColumn(fileName, columnMetadata);
      case CATEGORY:
        return readCategoryColumn(fileName, columnMetadata);
      case SHORT_INT:
        return readShortColumn(fileName, columnMetadata);
      case LONG_INT:
        return readLongColumn(fileName, columnMetadata);
      default:
        throw new RuntimeException("Unhandled column type writing columns");
    }
  }

  public static FloatColumn readFloatColumn(String fileName, ColumnMetadata metadata) throws IOException {
    FloatColumn floats = new FloatColumn(metadata);
    try (FileInputStream fis = new FileInputStream(fileName);
         SnappyFramedInputStream sis = new SnappyFramedInputStream(fis, true);
         DataInputStream dis = new DataInputStream(sis)) {
      boolean EOF = false;
      while (!EOF) {
        try {
          float cell = dis.readFloat();
          floats.add(cell);
        } catch (EOFException e) {
          EOF = true;
        }
      }
    }
    return floats;
  }

  public static IntColumn readIntColumn(String fileName, ColumnMetadata metadata) throws IOException {
    IntColumn ints = new IntColumn(metadata);
    try (FileInputStream fis = new FileInputStream(fileName);
         SnappyFramedInputStream sis = new SnappyFramedInputStream(fis, true);
         DataInputStream dis = new DataInputStream(sis)) {
      boolean EOF = false;
      while (!EOF) {
        try {
          ints.add(dis.readInt());
        } catch (EOFException e) {
          EOF = true;
        }
      }
    }
    return ints;
  }

  public static ShortColumn readShortColumn(String fileName, ColumnMetadata metadata) throws IOException {
    ShortColumn ints = new ShortColumn(metadata);
    try (FileInputStream fis = new FileInputStream(fileName);
         SnappyFramedInputStream sis = new SnappyFramedInputStream(fis, true);
         DataInputStream dis = new DataInputStream(sis)) {
      boolean EOF = false;
      while (!EOF) {
        try {
          ints.add(dis.readShort());
        } catch (EOFException e) {
          EOF = true;
        }
      }
    }
    return ints;
  }

  public static LongColumn readLongColumn(String fileName, ColumnMetadata metadata) throws IOException {
    LongColumn ints = new LongColumn(metadata);
    try (FileInputStream fis = new FileInputStream(fileName);
         SnappyFramedInputStream sis = new SnappyFramedInputStream(fis, true);
         DataInputStream dis = new DataInputStream(sis)) {
      boolean EOF = false;
      while (!EOF) {
        try {
          ints.add(dis.readLong());
        } catch (EOFException e) {
          EOF = true;
        }
      }
    }
    return ints;
  }

  public static DateColumn readLocalDateColumn(String fileName, ColumnMetadata metadata) throws IOException {
    DateColumn dates = new DateColumn(metadata);
    try (FileInputStream fis = new FileInputStream(fileName);
         SnappyFramedInputStream sis = new SnappyFramedInputStream(fis, true);
         DataInputStream dis = new DataInputStream(sis)) {
      boolean EOF = false;
      while (!EOF) {
        try {
          int cell = dis.readInt();
          dates.add(cell);
        } catch (EOFException e) {
          EOF = true;
        }
      }
    }
    return dates;
  }

  public static DateTimeColumn readLocalDateTimeColumn(String fileName, ColumnMetadata metadata) throws
      IOException {
    DateTimeColumn dates = new DateTimeColumn(metadata);
    try (FileInputStream fis = new FileInputStream(fileName);
         SnappyFramedInputStream sis = new SnappyFramedInputStream(fis, true);
         DataInputStream dis = new DataInputStream(sis)) {
      boolean EOF = false;
      while (!EOF) {
        try {
          long cell = dis.readLong();
          dates.add(cell);
        } catch (EOFException e) {
          EOF = true;
        }
      }
    }
    return dates;
  }

  public static TimeColumn readLocalTimeColumn(String fileName, ColumnMetadata metadata) throws IOException {
    TimeColumn times = new TimeColumn(metadata);
    try (FileInputStream fis = new FileInputStream(fileName);
         SnappyFramedInputStream sis = new SnappyFramedInputStream(fis, true);
         DataInputStream dis = new DataInputStream(sis)) {
      boolean EOF = false;
      while (!EOF) {
        try {
          int cell = dis.readInt();
          times.add(cell);
        } catch (EOFException e) {
          EOF = true;
        }
      }
    }
    return times;
  }

  public static CategoryColumn readCategoryColumn(String fileName, ColumnMetadata metadata) throws IOException {
    CategoryColumn stringColumn = new CategoryColumn(metadata);
    try (FileInputStream fis = new FileInputStream(fileName);
         SnappyFramedInputStream sis = new SnappyFramedInputStream(fis, true);
         DataInputStream dis = new DataInputStream(sis)) {

      int stringCount = dis.readInt();

      int j = 0;
      while (j < stringCount) {
        stringColumn.dictionaryMap().put(j, dis.readUTF());
        j++;
      }

      int size = metadata.getSize();
      for (int i = 0; i < size; i++) {
        stringColumn.data().add(dis.readInt());
      }
    }
    return stringColumn;
  }

  public static BooleanColumn readBooleanColumn(String fileName, ColumnMetadata metadata) throws IOException {
    BooleanColumn bools = new BooleanColumn(metadata);
    try (FileInputStream fis = new FileInputStream(fileName);
         SnappyFramedInputStream sis = new SnappyFramedInputStream(fis, true);
         DataInputStream dis = new DataInputStream(sis)) {
      boolean EOF = false;
      while (!EOF) {
        try {
          boolean cell = dis.readBoolean();
          bools.add(cell);
        } catch (EOFException e) {
          EOF = true;
        }
      }
    }
    return bools;
  }

  /**
   * Saves the data from the given table in the location specified by folderName. Within that folder each table has
   * its own sub-folder, whose name is based on the name of the table.
   * <p>
   * NOTE: If you store a table with the same name in the same folder. The data in that folder will be over-written.
   * <p>
   * The storage format is the tablesaw compressed column-oriented format, which consists of a set of file in a folder.
   * The name of the folder is based on the name of the table.
   *
   * @param folderName The location of the table (for example: "mytables")
   * @param table      The table to be saved
   * @return The path and name of the table
   * @throws IOException
   */
  public static String saveTable(String folderName, Relation table) throws IOException {

    ExecutorService executorService = Executors.newFixedThreadPool(10);
    CompletionService writerCompletionService = new ExecutorCompletionService<>(executorService);

    String name = table.name();
    name = WHITE_SPACE_PATTERN.matcher(name).replaceAll(""); // remove whitespace from the table name
    name = SEPARATOR_PATTERN.matcher(name).replaceAll("_"); // remove path separators from the table name

    String storageFolder = folderName + separator() + name + '.' + FILE_EXTENSION;

    Path path = Paths.get(storageFolder);

    if (!Files.exists(path)) {
      try {
        Files.createDirectories(path);
      } catch (IOException e) {
        e.printStackTrace();
      }
    }

    writeTableMetadata(path.toString() + separator() + "Metadata.json", table);

    try {
      for (Column column : table.columns()) {
        writerCompletionService.submit(() -> {
          Path columnPath = path.resolve(column.id());
          writeColumn(columnPath.toString(), column);
          return null;
        });
      }
      for (int i = 0; i < table.columnCount(); i++) {
        Future future = writerCompletionService.take();
        future.get();
      }
    } catch (InterruptedException | ExecutionException e) {
      throw new RuntimeException(e);
    }
    executorService.shutdown();
    return storageFolder;
  }

  private static void writeColumn(String fileName, Column column) {
    try {
      switch (column.type()) {
        case FLOAT:
          writeColumn(fileName, (FloatColumn) column);
          break;
        case INTEGER:
          writeColumn(fileName, (IntColumn) column);
          break;
        case BOOLEAN:
          writeColumn(fileName, (BooleanColumn) column);
          break;
        case LOCAL_DATE:
          writeColumn(fileName, (DateColumn) column);
          break;
        case LOCAL_TIME:
          writeColumn(fileName, (TimeColumn) column);
          break;
        case LOCAL_DATE_TIME:
          writeColumn(fileName, (DateTimeColumn) column);
          break;
        case CATEGORY:
          writeColumn(fileName, (CategoryColumn) column);
          break;
        case SHORT_INT:
          writeColumn(fileName, (ShortColumn) column);
          break;
        case LONG_INT:
          writeColumn(fileName, (LongColumn) column);
          break;
        default:
          throw new RuntimeException("Unhandled column type writing columns");
      }
    } catch (IOException ex) {
      throw new RuntimeException("IOException writing to file");
    }
  }

  public static void writeColumn(String fileName, FloatColumn column) throws IOException {
    try (FileOutputStream fos = new FileOutputStream(fileName);
         SnappyFramedOutputStream sos = new SnappyFramedOutputStream(fos);
         DataOutputStream dos = new DataOutputStream(sos)) {
      int i = 0;
      for (float d : column) {
        dos.writeFloat(d);
        if (i % FLUSH_AFTER_ITERATIONS == 0) {
          dos.flush();
        }
        i++;
      }
      dos.flush();
    }
  }

  /**
   * Writes out the values of the category column encoded as ints to minimize the time required for subsequent reads
   * <p>
   * The files are written Strings first, then the ints that encode them so they can be read in the opposite order
   *
   * @throws IOException
   */
  public static void writeColumn(String fileName, CategoryColumn column) throws IOException {

    int categoryCount = column.dictionaryMap().size();
    try (FileOutputStream fos = new FileOutputStream(fileName);
         SnappyFramedOutputStream sos = new SnappyFramedOutputStream(fos);
         DataOutputStream dos = new DataOutputStream(sos)) {

      dos.writeInt(categoryCount);
      // write the strings
      SortedSet<Integer> keys = new TreeSet<>(column.dictionaryMap().keyToValueMap().keySet());
      for (int key : keys) {
        dos.writeUTF(column.dictionaryMap().get(key));
      }
      dos.flush();

      // write the integer values that represent the strings
      int i = 0;
      for (int d : column.data()) {
        dos.writeInt(d);
        if (i % FLUSH_AFTER_ITERATIONS == 0) {
          dos.flush();
        }
        i++;
      }
    }
  }

  //TODO(lwhite): saveTable the column using integer compression
  public static void writeColumn(String fileName, IntColumn column) throws IOException {
    try (FileOutputStream fos = new FileOutputStream(fileName);
         SnappyFramedOutputStream sos = new SnappyFramedOutputStream(fos);
         DataOutputStream dos = new DataOutputStream(sos)) {
      int i = 0;
      for (int d : column.data()) {
        dos.writeInt(d);
        if (i % FLUSH_AFTER_ITERATIONS == 0) {
          dos.flush();
        }
        i++;
      }
      dos.flush();
    }
  }

  public static void writeColumn(String fileName, ShortColumn column) throws IOException {
    try (FileOutputStream fos = new FileOutputStream(fileName);
         SnappyFramedOutputStream sos = new SnappyFramedOutputStream(fos);
         DataOutputStream dos = new DataOutputStream(sos)) {
      int i = 0;
      for (short d : column) {
        dos.writeShort(d);
        if (i % FLUSH_AFTER_ITERATIONS == 0) {
          dos.flush();
        }
        i++;
      }
      dos.flush();
    }
  }

  public static void writeColumn(String fileName, LongColumn column) throws IOException {
    try (FileOutputStream fos = new FileOutputStream(fileName);
         SnappyFramedOutputStream sos = new SnappyFramedOutputStream(fos);
         DataOutputStream dos = new DataOutputStream(sos)) {
      int i = 0;
      for (long d : column) {
        dos.writeLong(d);
        if (i % FLUSH_AFTER_ITERATIONS == 0) {
          dos.flush();
        }
        i++;
      }
      dos.flush();
    }
  }

  //TODO(lwhite): saveTable the column using integer compression
  public static void writeColumn(String fileName, DateColumn column) throws IOException {
    try (FileOutputStream fos = new FileOutputStream(fileName);
         SnappyFramedOutputStream sos = new SnappyFramedOutputStream(fos);
         DataOutputStream dos = new DataOutputStream(sos)) {
      int i = 0;
      for (int d : column.data()) {
        dos.writeInt(d);
        if (i % FLUSH_AFTER_ITERATIONS == 0) {
          dos.flush();
        }
        i++;
      }
      dos.flush();
    }
  }

  public static void writeColumn(String fileName, DateTimeColumn column) throws IOException {
    try (FileOutputStream fos = new FileOutputStream(fileName);
         SnappyFramedOutputStream sos = new SnappyFramedOutputStream(fos);
         DataOutputStream dos = new DataOutputStream(sos)) {
      int i = 0;
      for (long d : column.data()) {
        dos.writeLong(d);
        if (i % FLUSH_AFTER_ITERATIONS == 0) {
          dos.flush();
        }
        i++;
      }
      dos.flush();
    }
  }

  //TODO(lwhite): saveTable the column using integer compression
  public static void writeColumn(String fileName, TimeColumn column) throws IOException {
    try (FileOutputStream fos = new FileOutputStream(fileName);
         SnappyFramedOutputStream sos = new SnappyFramedOutputStream(fos);
         DataOutputStream dos = new DataOutputStream(sos)) {
      int i = 0;
      for (int d : column.data()) {
        dos.writeInt(d);
        if (i % FLUSH_AFTER_ITERATIONS == 0) {
          dos.flush();
        }
        i++;
      }
      dos.flush();
    }
  }

  //TODO(lwhite): saveTable the column using compressed bitmap
  public static void writeColumn(String fileName, BooleanColumn column) throws IOException {
    try (FileOutputStream fos = new FileOutputStream(fileName);
         SnappyFramedOutputStream sos = new SnappyFramedOutputStream(fos);
         DataOutputStream dos = new DataOutputStream(sos)) {
      for (int i = 0; i < column.size(); i++) {
        boolean value = column.get(i);
        dos.writeBoolean(value);
        if (i % FLUSH_AFTER_ITERATIONS == 0) {
          dos.flush();
        }
      }
      dos.flush();
    }
  }

  /**
   * Writes out a json-formatted representation of the given {@code table}'s metadata to the given {@code file}
   *
   * @param fileName Expected to be fully specified
   * @throws IOException if the file can not be read
   */
  public static void writeTableMetadata(String fileName, Relation table) throws IOException {
    File myFile = Paths.get(fileName).toFile();
    myFile.createNewFile();
    try (FileOutputStream fOut = new FileOutputStream(myFile);
         OutputStreamWriter myOutWriter = new OutputStreamWriter(fOut)) {
      myOutWriter.append(new TableMetadata(table).toJson());
    }
  }

  /**
   * Reads in a json-formatted file and creates a TableMetadata instance from it. Files are expected to be in
   * the format provided by TableMetadata}
   *
   * @param fileName Expected to be fully specified
   * @throws IOException if the file can not be read
   */
  public static TableMetadata readTableMetadata(String fileName) throws IOException {

    byte[] encoded = Files.readAllBytes(Paths.get(fileName));
    return TableMetadata.fromJson(new String(encoded, StandardCharsets.UTF_8));
  }
}
