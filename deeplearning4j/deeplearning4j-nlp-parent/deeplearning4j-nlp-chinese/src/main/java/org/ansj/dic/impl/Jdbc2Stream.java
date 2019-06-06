package org.ansj.dic.impl;

import org.ansj.dic.PathToStream;
import org.ansj.exception.LibraryException;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;

/**
 * jdbc:mysql://192.168.10.103:3306/infcn_mss?useUnicode=true&characterEncoding=utf-8&zeroDateTimeBehavior=convertToNull|username|password|select name as name,nature,freq from dic where type=1
 *
 * @author ansj
 */
public class Jdbc2Stream extends PathToStream {

    private static final byte[] TAB = "\t".getBytes();

    private static final byte[] LINE = "\n".getBytes();

    static {
        String[] drivers = {"org.h2.Driver", "com.ibm.db2.jcc.DB2Driver", "org.hsqldb.jdbcDriver",
                        "org.gjt.mm.mysql.Driver", "oracle.jdbc.OracleDriver", "org.postgresql.Driver",
                        "net.sourceforge.jtds.jdbc.Driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver",
                        "org.sqlite.JDBC", "com.mysql.jdbc.Driver"};
        for (String driverClassName : drivers) {
            try {
                try {
                    Thread.currentThread().getContextClassLoader().loadClass(driverClassName);
                } catch (ClassNotFoundException e) {
                    Class.forName(driverClassName);
                }
            } catch (Throwable e) {
            }
        }
    }

    @Override
    public InputStream toStream(String path) {
        path = path.substring(7);

        String[] split = path.split("\\|");

        String jdbc = split[0];

        String username = split[1];

        String password = split[2];

        String sqlStr = split[3];

        String logStr = jdbc + "|" + username + "|********|" + sqlStr;

        try (Connection conn = DriverManager.getConnection(jdbc, username, password);
                        PreparedStatement statement = conn.prepareStatement(sqlStr);
                        ResultSet rs = statement.executeQuery();
                        ByteArrayOutputStream baos = new ByteArrayOutputStream(100 * 1024)) {

            int i, count;
            while (rs.next()) {
                for (i = 1, count = rs.getMetaData().getColumnCount(); i < count; ++i) {
                    baos.write(String.valueOf(rs.getObject(i)).getBytes());
                    baos.write(TAB);
                }
                baos.write(String.valueOf(rs.getObject(count)).getBytes());
                baos.write(LINE);
            }

            return new ByteArrayInputStream(baos.toByteArray());
        } catch (Exception e) {
            throw new LibraryException("err to load by jdbc " + logStr);
        }
    }

    public static String encryption(String path) {

        String[] split = path.split("\\|");

        String jdbc = split[0];

        String username = split[1];

        String password = split[2];

        String sqlStr = split[3];

        return jdbc + "|" + username + "|********|" + sqlStr;
    }
}
