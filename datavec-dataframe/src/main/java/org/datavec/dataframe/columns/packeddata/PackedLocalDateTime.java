package org.datavec.dataframe.columns.packeddata;

import com.google.common.base.Strings;
import com.google.common.primitives.Ints;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.Month;
import java.time.ZoneId;
import java.time.chrono.IsoChronology;
import java.time.temporal.ChronoField;
import java.util.Date;

import static org.datavec.dataframe.columns.packeddata.PackedLocalDate.asLocalDate;

/**
 * A short localdatetime packed into a single long value. The long is comprised of an int for the date and an int
 * for the time
 * <p>
 * The bytes are packed into the date int as:
 * First two bytes: short (year)
 * next byte (month of year)
 * last byte (day of month)
 * <p>
 * The bytes are packed into the time int as
 * First byte: hourOfDay
 * next byte: minuteOfHour
 * last two bytes (short): millisecond of minute
 * <p>
 * Storing the millisecond of minute in an short requires that we treat the short as if it were unsigned. Unfortunately,
 * Neither Java nor Guava provide unsigned short support so we use char, which is a 16-bit unsigned int to
 * store values of up to 60,000 milliseconds (60 secs * 1000)
 */
public class PackedLocalDateTime {

  public static byte getDayOfMonth(long date) {
    return (byte) date(date);  // last byte
  }

  public static short getYear(int date) {
    // get first two bytes, then convert to a short
    byte byte1 = (byte) (date >> 24);
    byte byte2 = (byte) (date >> 16);
    return (short) ((byte2 << 8) + (byte1 & 0xFF));
  }

  public static short getYear(long dateTime) {
    return getYear(date(dateTime));
  }

  public static LocalDateTime asLocalDateTime(long dateTime) {
    if (dateTime == Long.MIN_VALUE) {
      return null;
    }
    int date = date(dateTime);
    int time = time(dateTime);

    return LocalDateTime.of(asLocalDate(date), PackedLocalTime.asLocalTime(time));
  }

  public static byte getMonthValue(long dateTime) {
    int date = date(dateTime);
    return (byte) (date >> 8);
  }

  public static long pack(LocalDate date, LocalTime time) {
    int d = PackedLocalDate.pack(date);
    int t = PackedLocalTime.pack(time);
    return (((long) d) << 32) | (t & 0xffffffffL);
  }

  public static long pack(LocalDateTime dateTime) {
    LocalDate date = dateTime.toLocalDate();
    LocalTime time = dateTime.toLocalTime();
    int d = PackedLocalDate.pack(date);
    int t = PackedLocalTime.pack(time);
    return (((long) d) << 32) | (t & 0xffffffffL);
  }

  public static long pack(Date date) {
    return pack(date.toInstant().atZone(ZoneId.systemDefault()).toLocalDateTime());
  }

  public static long pack(short yr, byte m, byte d, byte hr, byte min, byte s, byte n) {
    byte byte1 = (byte) yr;
    byte byte2 = (byte) ((yr >> 8) & 0xff);
    int date = Ints.fromBytes(
        byte1,
        byte2,
        m,
        d);

    int time = Ints.fromBytes(hr, min, s, n);

    return (((long) date) << 32) | (time & 0xffffffffL);
  }

  public static int date(long packedDateTIme) {
    return (int) (packedDateTIme >> 32);
  }

  public static int time(long packedDateTIme) {
    return (int) packedDateTIme;
  }

  public static String toString(long dateTime) {
    if (dateTime == Long.MIN_VALUE) {
      return "NA";
    }
    int date = date(dateTime);
    int time = time(dateTime);

    // get first two bytes, then each of the other two
    byte yearByte1 = (byte) (date >> 24);
    byte yearByte2 = (byte) (date >> 16);

    return
        "" + (short) ((yearByte2 << 8) + (yearByte1 & 0xFF))
            + "-"
            + Strings.padStart(Byte.toString((byte) (date >> 8)), 2, '0')
            + "-"
            + Strings.padStart(Byte.toString((byte) date), 2, '0')
            + "T"
            + Strings.padStart(Byte.toString(PackedLocalTime.getHour(time)), 2, '0')
            + ":"
            + Strings.padStart(Byte.toString(PackedLocalTime.getMinute(time)), 2, '0')
            + ":"
            + Strings.padStart(Byte.toString(PackedLocalTime.getSecond(time)), 2, '0')
            + "."
            + Strings.padStart(String.valueOf(PackedLocalTime.getMilliseconds(time)), 3, '0');
  }

  public static int getDayOfYear(long packedDateTime) {

    return getMonth(packedDateTime).firstDayOfYear(isLeapYear(packedDateTime)) + getDayOfMonth(packedDateTime) - 1;
  }

  public static boolean isLeapYear(long packedDateTime) {
    return IsoChronology.INSTANCE.isLeapYear(getYear(packedDateTime));
  }

  public static Month getMonth(long packedDateTime) {
    return Month.of(getMonthValue(packedDateTime));
  }

  public static int lengthOfMonth(long packedDateTime) {
    switch (getMonthValue(packedDateTime)) {
      case 2:
        return (isLeapYear(packedDateTime) ? 29 : 28);
      case 4:
      case 6:
      case 9:
      case 11:
        return 30;
      default:
        return 31;
    }
  }

  public int lengthOfYear(long packedDateTime) {
    return (isLeapYear(packedDateTime) ? 366 : 365);
  }

  public static DayOfWeek getDayOfWeek(long packedDateTime) {
    int date = PackedLocalDateTime.date(packedDateTime);
    return PackedLocalDate.getDayOfWeek(date);
  }

  private static long toEpochDay(long packedDateTime) {
    return PackedLocalDate.toEpochDay(date(packedDateTime));
  }

  public static boolean isInQ1(long packedDateTime) {
    Month month = getMonth(packedDateTime);
    return month == Month.JANUARY ||
        month == Month.FEBRUARY ||
        month == Month.MARCH;
  }

  public static boolean isInQ2(long packedDateTime) {
    Month month = getMonth(packedDateTime);
    return month == Month.APRIL ||
        month == Month.MAY ||
        month == Month.JUNE;
  }

  public static boolean isInQ3(long packedDateTime) {
    Month month = getMonth(packedDateTime);
    return month == Month.JULY ||
        month == Month.AUGUST ||
        month == Month.SEPTEMBER;
  }

  public static boolean isInQ4(long packedDateTime) {
    Month month = getMonth(packedDateTime);
    return month == Month.OCTOBER ||
        month == Month.NOVEMBER ||
        month == Month.DECEMBER;
  }

  public static boolean isAfter(long packedDateTime, long value) {
    return packedDateTime > value;
  }

  public static boolean isBefore(long packedDateTime, long value) {
    return packedDateTime < value;
  }

  public static boolean isSunday(long packedDateTime) {
    DayOfWeek dayOfWeek = getDayOfWeek(packedDateTime);
    return dayOfWeek == DayOfWeek.SUNDAY;
  }

  public static boolean isMonday(long packedDateTime) {
    DayOfWeek dayOfWeek = getDayOfWeek(packedDateTime);
    return dayOfWeek == DayOfWeek.MONDAY;
  }

  public static boolean isTuesday(long packedDateTime) {
    DayOfWeek dayOfWeek = getDayOfWeek(packedDateTime);
    return dayOfWeek == DayOfWeek.TUESDAY;
  }

  public static boolean isWednesday(long packedDateTime) {
    DayOfWeek dayOfWeek = getDayOfWeek(packedDateTime);
    return dayOfWeek == DayOfWeek.WEDNESDAY;
  }

  public static boolean isThursday(long packedDateTime) {
    DayOfWeek dayOfWeek = getDayOfWeek(packedDateTime);
    return dayOfWeek == DayOfWeek.THURSDAY;
  }

  public static boolean isFriday(long packedDateTime) {
    DayOfWeek dayOfWeek = getDayOfWeek(packedDateTime);
    return dayOfWeek == DayOfWeek.FRIDAY;
  }

  public static boolean isSaturday(long packedDateTime) {
    DayOfWeek dayOfWeek = getDayOfWeek(packedDateTime);
    return dayOfWeek == DayOfWeek.SATURDAY;
  }

  public static boolean isFirstDayOfMonth(long packedDateTime) {
    return getDayOfMonth(packedDateTime) == 1;
  }

  public static boolean isInJanuary(long packedDateTime) {
    return getMonth(packedDateTime) == Month.JANUARY;
  }

  public static boolean isInFebruary(long packedDateTime) {
    return getMonth(packedDateTime) == Month.FEBRUARY;
  }

  public static boolean isInMarch(long packedDateTime) {
    return getMonth(packedDateTime) == Month.MARCH;
  }

  public static boolean isInApril(long packedDateTime) {
    return getMonth(packedDateTime) == Month.APRIL;
  }

  public static boolean isInMay(long packedDateTime) {
    return getMonth(packedDateTime) == Month.MAY;
  }

  public static boolean isInJune(long packedDateTime) {
    return getMonth(packedDateTime) == Month.JUNE;
  }

  public static boolean isInJuly(long packedDateTime) {
    return getMonth(packedDateTime) == Month.JULY;
  }

  public static boolean isInAugust(long packedDateTime) {
    return getMonth(packedDateTime) == Month.AUGUST;
  }

  public static boolean isInSeptember(long packedDateTime) {
    return getMonth(packedDateTime) == Month.SEPTEMBER;
  }

  public static boolean isInOctober(long packedDateTime) {
    return getMonth(packedDateTime) == Month.OCTOBER;
  }

  public static boolean isInNovember(long packedDateTime) {
    return getMonth(packedDateTime) == Month.NOVEMBER;
  }

  public static boolean isInDecember(long packedDateTime) {
    return getMonth(packedDateTime) == Month.DECEMBER;
  }

  public static boolean isLastDayOfMonth(long packedDateTime) {
    return getDayOfMonth(packedDateTime) == lengthOfMonth(packedDateTime);
  }

  public static boolean isInYear(long packedDateTime, int year) {
    return getYear(packedDateTime) == year;
  }

  public static boolean isMidnight(long packedDateTime) {
    return PackedLocalTime.isMidnight(time(packedDateTime));
  }

  public static boolean isNoon(long packedDateTime) {
    return PackedLocalTime.isNoon(time(packedDateTime));
  }

  /**
   * Returns true if the time is in the AM or "before noon".
   * Note: we follow the convention that 12:00 NOON is PM and 12 MIDNIGHT is AM
   */
  public static boolean AM(long packedDateTime) {
    return PackedLocalTime.AM(time(packedDateTime));
  }

  /**
   * Returns true if the time is in the PM or "after noon".
   * Note: we follow the convention that 12:00 NOON is PM and 12 MIDNIGHT is AM
   */
  public static boolean PM(long packedDateTime) {
    return PackedLocalTime.PM(time(packedDateTime));
  }

  public static int getMinuteOfDay(long packedLocalDateTime) {
    return getHour(packedLocalDateTime) * 60 + getMinute(packedLocalDateTime);
  }

  public static byte getSecond(int packedLocalDateTime) {
    return (byte) (getMillisecondOfMinute(packedLocalDateTime) / 1000);
  }

  public static byte getHour(long packedLocalDateTime) {
    return PackedLocalTime.getHour(time(packedLocalDateTime));
  }

  public static byte getMinute(long packedLocalDateTime) {
    return PackedLocalTime.getMinute(time(packedLocalDateTime));
  }

  public static byte getSecond(long packedLocalDateTime) {
    return PackedLocalTime.getSecond(time(packedLocalDateTime));
  }

  public static int getSecondOfDay(long packedLocalDateTime) {
    return PackedLocalTime.getSecondOfDay(time(packedLocalDateTime));
  }

  public static short getMillisecondOfMinute(long packedLocalDateTime) {
    LocalDateTime localDateTime = LocalDateTime.now();
    short total = (short) localDateTime.get(ChronoField.MILLI_OF_SECOND);
    total += localDateTime.getSecond() * 1000;
    return total;
  }

  public static long getMillisecondOfDay(long packedLocalDateTime) {
    LocalDateTime localDateTime = PackedLocalDateTime.asLocalDateTime(packedLocalDateTime);
    long total = (long) localDateTime.get(ChronoField.MILLI_OF_SECOND);
    total += localDateTime.getSecond() * 1000;
    total += localDateTime.getMinute() * 60 * 1000;
    total += localDateTime.getHour() * 60 * 60 * 1000;
    return total;
  }

  public static long create(int date, int time) {
    return (((long) date) << 32) | (time & 0xffffffffL);

  }
}
