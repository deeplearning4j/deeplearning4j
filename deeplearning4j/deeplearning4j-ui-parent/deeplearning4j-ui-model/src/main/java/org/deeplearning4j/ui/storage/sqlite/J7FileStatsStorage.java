/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.ui.storage.sqlite;

import it.unimi.dsi.fastutil.longs.LongArrayList;
import lombok.NonNull;
import org.deeplearning4j.api.storage.*;
import org.nd4j.linalg.primitives.Pair;

import java.io.*;
import java.sql.*;
import java.util.*;

/**
 * A Java 7 compatible file-based {@link StatsStorage} implementation, based on SQLite.
 * Note: Where possible, use {@link org.deeplearning4j.ui.storage.FileStatsStorage} which should be faster (is based
 * on MapDB).
 * Obviously, the storage formats for J7FileStatsStorage and {@link org.deeplearning4j.ui.storage.FileStatsStorage} are
 * incompatible.
 *
 * @author Alex Black
 */
public class J7FileStatsStorage implements StatsStorage {

    private static final String TABLE_NAME_METADATA = "StorageMetaData";
    private static final String TABLE_NAME_STATIC_INFO = "StaticInfo";
    private static final String TABLE_NAME_UPDATES = "Updates";

    private static final String INSERT_META_SQL = "INSERT OR REPLACE INTO " + TABLE_NAME_METADATA
                    + " (SessionID, TypeID, ObjectClass, ObjectBytes) VALUES ( ?, ?, ?, ? );";
    private static final String INSERT_STATIC_SQL = "INSERT OR REPLACE INTO " + TABLE_NAME_STATIC_INFO
                    + " (SessionID, TypeID, WorkerID, ObjectClass, ObjectBytes) VALUES ( ?, ?, ?, ?, ? );";
    private static final String INSERT_UPDATE_SQL = "INSERT OR REPLACE INTO " + TABLE_NAME_UPDATES
                    + " (SessionID, TypeID, WorkerID, Timestamp, ObjectClass, ObjectBytes) VALUES ( ?, ?, ?, ?, ?, ? );";

    private final File file;
    private final Connection connection;
    private List<StatsStorageListener> listeners = new ArrayList<>();

    /**
     * @param file Storage location for the stats
     */
    public J7FileStatsStorage(@NonNull File file) {
        this.file = file;

        try {
            connection = DriverManager.getConnection("jdbc:sqlite:" + file.getAbsolutePath());
        } catch (Exception e) {
            throw new RuntimeException("Error ninializing J7FileStatsStorage instance", e);
        }

        try {
            initializeTables();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    private void initializeTables() throws SQLException {

        //Need tables for:
        //(a) Metadata  -> session ID and type ID; class; StorageMetaData as a binary BLOB
        //(b) Static info -> session ID, type ID, worker ID, persistable class, persistable bytes
        //(c) Update info -> session ID, type ID, worker ID, timestamp, update class, update bytes

        //First: check if tables exist
        DatabaseMetaData meta = connection.getMetaData();
        ResultSet rs = meta.getTables(null, null, "%", null);
        boolean hasStorageMetaDataTable = false;
        boolean hasStaticInfoTable = false;
        boolean hasUpdatesTable = false;
        while (rs.next()) {
            //3rd value: table name - http://docs.oracle.com/javase/6/docs/api/java/sql/DatabaseMetaData.html#getTables%28java.lang.String,%20java.lang.String,%20java.lang.String,%20java.lang.String[]%29
            String name = rs.getString(3);
            if (TABLE_NAME_METADATA.equals(name))
                hasStorageMetaDataTable = true;
            else if (TABLE_NAME_STATIC_INFO.equals(name))
                hasStaticInfoTable = true;
            else if (TABLE_NAME_UPDATES.equals(name))
                hasUpdatesTable = true;
        }


        Statement statement = connection.createStatement();

        if (!hasStorageMetaDataTable) {
            statement.executeUpdate("CREATE TABLE " + TABLE_NAME_METADATA + " (" + "SessionID TEXT NOT NULL, "
                            + "TypeID TEXT NOT NULL, " + "ObjectClass TEXT NOT NULL, " + "ObjectBytes BLOB NOT NULL, "
                            + "PRIMARY KEY ( SessionID, TypeID )" + ");");
        }

        if (!hasStaticInfoTable) {
            statement.executeUpdate("CREATE TABLE " + TABLE_NAME_STATIC_INFO + " (" + "SessionID TEXT NOT NULL, "
                            + "TypeID TEXT NOT NULL, " + "WorkerID TEXT NOT NULL, " + "ObjectClass TEXT NOT NULL, "
                            + "ObjectBytes BLOB NOT NULL, " + "PRIMARY KEY ( SessionID, TypeID, WorkerID )" + ");");
        }

        if (!hasUpdatesTable) {
            statement.executeUpdate("CREATE TABLE " + TABLE_NAME_UPDATES + " (" + "SessionID TEXT NOT NULL, "
                            + "TypeID TEXT NOT NULL, " + "WorkerID TEXT NOT NULL, " + "Timestamp INTEGER NOT NULL, "
                            + "ObjectClass TEXT NOT NULL, " + "ObjectBytes BLOB NOT NULL, "
                            + "PRIMARY KEY ( SessionID, TypeID, WorkerID, Timestamp )" + ");");
        }

        statement.close();

    }

    private static Pair<String, byte[]> serializeForDB(Object object) {
        String classStr = object.getClass().getName();
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
                        ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(object);
            oos.close();
            byte[] bytes = baos.toByteArray();
            return new Pair<>(classStr, bytes);
        } catch (IOException e) {
            throw new RuntimeException("Error serializing object for storage", e);
        }
    }

    private static <T> T deserialize(byte[] bytes) {
        try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(bytes))) {
            return (T) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    private <T> T queryAndGet(String sql, int columnIndex) {
        try (Statement statement = connection.createStatement()) {
            ResultSet rs = statement.executeQuery(sql);
            if (!rs.next())
                return null;
            byte[] bytes = rs.getBytes(columnIndex);
            return deserialize(bytes);
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    private List<String> selectDistinct(String columnName, boolean queryMeta, boolean queryStatic, boolean queryUpdates,
                    String conditionColumn, String conditionValue) {
        Set<String> unique = new HashSet<>();

        try (Statement statement = connection.createStatement()) {
            if (queryMeta) {
                queryHelper(statement, querySqlHelper(columnName, TABLE_NAME_METADATA, conditionColumn, conditionValue),
                                unique);
            }

            if (queryStatic) {
                queryHelper(statement,
                                querySqlHelper(columnName, TABLE_NAME_STATIC_INFO, conditionColumn, conditionValue),
                                unique);
            }

            if (queryUpdates) {
                queryHelper(statement, querySqlHelper(columnName, TABLE_NAME_UPDATES, conditionColumn, conditionValue),
                                unique);
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }

        return new ArrayList<>(unique);
    }

    private String querySqlHelper(String columnName, String table, String conditionColumn, String conditionValue) {
        String unique = "SELECT DISTINCT " + columnName + " FROM " + table;
        if (conditionColumn != null) {
            unique += " WHERE " + conditionColumn + " = '" + conditionValue + "'";
        }
        unique += ";";
        return unique;
    }

    private void queryHelper(Statement statement, String q, Set<String> unique) throws SQLException {
        ResultSet rs = statement.executeQuery(q);
        while (rs.next()) {
            String str = rs.getString(1);
            unique.add(str);
        }
    }

    protected List<StatsStorageEvent> checkStorageEvents(Persistable p) {
        if (listeners.isEmpty())
            return null;

        StatsStorageEvent newSID = null;
        StatsStorageEvent newTID = null;
        StatsStorageEvent newWID = null;

        String sid = p.getSessionID();
        String tid = p.getTypeID();
        String wid = p.getWorkerID();

        //Is this a new session ID? type ID? worker ID?

        //This is not the most efficient approach
        boolean isNewSID = false;
        boolean isNewTID = false;
        boolean isNewWID = false;
        if (!listSessionIDs().contains(sid)) {
            isNewSID = true;
            isNewTID = true;
            isNewWID = true;
        }

        if (!isNewTID && !listTypeIDsForSession(sid).contains(tid)) {
            isNewTID = true;
        }

        if (!isNewWID && !listWorkerIDsForSessionAndType(sid, tid).contains(wid)) {
            isNewWID = true;
        }

        if (isNewSID) {
            newSID = new StatsStorageEvent(this, StatsStorageListener.EventType.NewSessionID, p.getSessionID(),
                            p.getTypeID(), p.getWorkerID(), p.getTimeStamp());
        }
        if (isNewTID) {
            newTID = new StatsStorageEvent(this, StatsStorageListener.EventType.NewTypeID, p.getSessionID(),
                            p.getTypeID(), p.getWorkerID(), p.getTimeStamp());
        }
        if (isNewWID) {
            newWID = new StatsStorageEvent(this, StatsStorageListener.EventType.NewWorkerID, p.getSessionID(),
                            p.getTypeID(), p.getWorkerID(), p.getTimeStamp());
        }

        if (!isNewSID && !isNewTID && !isNewWID)
            return null;
        List<StatsStorageEvent> sses = new ArrayList<>(3);
        if (newSID != null)
            sses.add(newSID);
        if (newTID != null)
            sses.add(newTID);
        if (newWID != null)
            sses.add(newWID);
        return sses;
    }

    @Override
    public void putStorageMetaData(StorageMetaData storageMetaData) {
        putStorageMetaData(Collections.singletonList(storageMetaData));
    }

    @Override
    public void putStorageMetaData(Collection<? extends StorageMetaData> collection) {
        List<StatsStorageEvent> sses = null;
        try {
            PreparedStatement ps = connection.prepareStatement(INSERT_META_SQL);

            for (StorageMetaData storageMetaData : collection) {
                List<StatsStorageEvent> ssesTemp = checkStorageEvents(storageMetaData);
                if (ssesTemp != null) {
                    if (sses == null)
                        sses = ssesTemp;
                    else
                        sses.addAll(ssesTemp);
                }

                if (!listeners.isEmpty()) {
                    StatsStorageEvent sse = new StatsStorageEvent(this, StatsStorageListener.EventType.PostMetaData,
                                    storageMetaData.getSessionID(), storageMetaData.getTypeID(),
                                    storageMetaData.getWorkerID(), storageMetaData.getTimeStamp());
                    if (sses == null)
                        sses = new ArrayList<>();
                    sses.add(sse);
                }


                //Normally we'd batch these... sqlite has an autocommit feature that messes up batching with .addBatch() and .executeUpdate()
                Pair<String, byte[]> p = serializeForDB(storageMetaData);

                ps.setString(1, storageMetaData.getSessionID());
                ps.setString(2, storageMetaData.getTypeID());
                ps.setString(3, p.getFirst());
                ps.setObject(4, p.getSecond());
                ps.executeUpdate();
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }

        notifyListeners(sses);
    }

    @Override
    public void putStaticInfo(Persistable staticInfo) {
        putStaticInfo(Collections.singletonList(staticInfo));
    }

    @Override
    public void putStaticInfo(Collection<? extends Persistable> collection) {
        List<StatsStorageEvent> sses = null;
        try {
            PreparedStatement ps = connection.prepareStatement(INSERT_STATIC_SQL);

            for (Persistable p : collection) {
                List<StatsStorageEvent> ssesTemp = checkStorageEvents(p);
                if (ssesTemp != null) {
                    if (sses == null)
                        sses = ssesTemp;
                    else
                        sses.addAll(ssesTemp);
                }

                if (!listeners.isEmpty()) {
                    StatsStorageEvent sse = new StatsStorageEvent(this, StatsStorageListener.EventType.PostStaticInfo,
                                    p.getSessionID(), p.getTypeID(), p.getWorkerID(), p.getTimeStamp());
                    if (sses == null)
                        sses = new ArrayList<>();
                    sses.add(sse);
                }

                //Normally we'd batch these... sqlite has an autocommit feature that messes up batching with .addBatch() and .executeUpdate()
                Pair<String, byte[]> pair = serializeForDB(p);

                ps.setString(1, p.getSessionID());
                ps.setString(2, p.getTypeID());
                ps.setString(3, p.getWorkerID());
                ps.setString(4, pair.getFirst());
                ps.setBytes(5, pair.getSecond());
                ps.executeUpdate();
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }

        notifyListeners(sses);
    }

    @Override
    public void putUpdate(Persistable update) {
        putUpdate(Collections.singletonList(update));
    }

    @Override
    public void putUpdate(Collection<? extends Persistable> collection) {
        List<StatsStorageEvent> sses = null;

        try {
            PreparedStatement ps = connection.prepareStatement(INSERT_UPDATE_SQL);

            for (Persistable p : collection) {
                List<StatsStorageEvent> ssesTemp = checkStorageEvents(p);
                if (ssesTemp != null) {
                    if (sses == null)
                        sses = ssesTemp;
                    else
                        sses.addAll(ssesTemp);
                }

                if (!listeners.isEmpty()) {
                    StatsStorageEvent sse = new StatsStorageEvent(this, StatsStorageListener.EventType.PostUpdate,
                                    p.getSessionID(), p.getTypeID(), p.getWorkerID(), p.getTimeStamp());
                    if (sses == null)
                        sses = new ArrayList<>();
                    sses.add(sse);
                }

                //Normally we'd batch these... sqlite has an autocommit feature that messes up batching with .addBatch() and .executeUpdate()
                Pair<String, byte[]> pair = serializeForDB(p);

                ps.setString(1, p.getSessionID());
                ps.setString(2, p.getTypeID());
                ps.setString(3, p.getWorkerID());
                ps.setLong(4, p.getTimeStamp());
                ps.setString(5, pair.getFirst());
                ps.setObject(6, pair.getSecond());
                ps.executeUpdate();
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }

        notifyListeners(sses);
    }

    @Override
    public void close() throws IOException {
        try {
            connection.close();
        } catch (Exception e) {
            throw new IOException(e);
        }
    }

    @Override
    public boolean isClosed() {
        try {
            return connection.isClosed();
        } catch (Exception e) {
            return true;
        }
    }

    @Override
    public List<String> listSessionIDs() {
        return selectDistinct("SessionID", true, true, false, null, null);
    }

    @Override
    public boolean sessionExists(String sessionID) {
        String existsMetaSQL = "SELECT 1 FROM " + TABLE_NAME_METADATA + " WHERE SessionID = '" + sessionID + "';";
        String existsStaticSQL = "SELECT 1 FROM " + TABLE_NAME_STATIC_INFO + " WHERE SessionID = '" + sessionID + "';";

        try (Statement statement = connection.createStatement()) {
            ResultSet rs = statement.executeQuery(existsMetaSQL);
            if (rs.next()) {
                return true;
            }

            rs = statement.executeQuery(existsStaticSQL);
            return rs.next();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Persistable getStaticInfo(String sessionID, String typeID, String workerID) {
        String selectStaticSQL = "SELECT ObjectBytes FROM " + TABLE_NAME_STATIC_INFO + " WHERE SessionID = '"
                        + sessionID + "' AND TypeID = '" + typeID + "' AND WorkerID = '" + workerID + "';";
        return queryAndGet(selectStaticSQL, 1);
    }

    @Override
    public List<Persistable> getAllStaticInfos(String sessionID, String typeID) {
        String selectStaticSQL = "SELECT * FROM " + TABLE_NAME_STATIC_INFO + " WHERE SessionID = '" + sessionID
                        + "' AND TypeID = '" + typeID + "';";
        try (Statement statement = connection.createStatement()) {
            ResultSet rs = statement.executeQuery(selectStaticSQL);
            List<Persistable> out = new ArrayList<>();
            while (rs.next()) {
                byte[] bytes = rs.getBytes(5);
                out.add((Persistable) deserialize(bytes));
            }
            return out;
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public List<String> listTypeIDsForSession(String sessionID) {
        return selectDistinct("TypeID", true, true, true, "SessionID", sessionID);
    }

    @Override
    public List<String> listWorkerIDsForSession(String sessionID) {
        return selectDistinct("WorkerID", false, true, true, "SessionID", sessionID);
    }

    @Override
    public List<String> listWorkerIDsForSessionAndType(String sessionID, String typeID) {
        String uniqueStatic = "SELECT DISTINCT WorkerID FROM " + TABLE_NAME_STATIC_INFO + " WHERE SessionID = '"
                        + sessionID + "' AND TypeID = '" + typeID + "';";
        String uniqueUpdates = "SELECT DISTINCT WorkerID FROM " + TABLE_NAME_UPDATES + " WHERE SessionID = '"
                        + sessionID + "' AND TypeID = '" + typeID + "';";

        Set<String> unique = new HashSet<>();
        try (Statement statement = connection.createStatement()) {
            ResultSet rs = statement.executeQuery(uniqueStatic);
            while (rs.next()) {
                String str = rs.getString(1);
                unique.add(str);
            }

            rs = statement.executeQuery(uniqueUpdates);
            while (rs.next()) {
                String str = rs.getString(1);
                unique.add(str);
            }

        } catch (SQLException e) {
            throw new RuntimeException(e);
        }

        return new ArrayList<>(unique);
    }

    @Override
    public int getNumUpdateRecordsFor(String sessionID) {
        String sql = "SELECT COUNT(*) FROM " + TABLE_NAME_UPDATES + " WHERE SessionID = '" + sessionID + "';";
        try (Statement statement = connection.createStatement()) {
            return statement.executeQuery(sql).getInt(1);
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int getNumUpdateRecordsFor(String sessionID, String typeID, String workerID) {
        String sql = "SELECT COUNT(*) FROM " + TABLE_NAME_UPDATES + " WHERE SessionID = '" + sessionID
                        + "' AND TypeID = '" + typeID + "' AND WorkerID = '" + workerID + "';";
        try (Statement statement = connection.createStatement()) {
            return statement.executeQuery(sql).getInt(1);
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Persistable getLatestUpdate(String sessionID, String typeID, String workerID) {
        String sql = "SELECT ObjectBytes FROM " + TABLE_NAME_UPDATES + " WHERE SessionID = '" + sessionID
                        + "' AND TypeID = '" + typeID + "' AND WorkerID = '" + workerID
                        + "' ORDER BY Timestamp DESC LIMIT 1;";
        return queryAndGet(sql, 1);
    }

    @Override
    public Persistable getUpdate(String sessionID, String typeId, String workerID, long timestamp) {
        String sql = "SELECT ObjectBytes FROM " + TABLE_NAME_UPDATES + " WHERE SessionID = '" + sessionID
                        + "' AND TypeID = '" + typeId + "' AND WorkerID = '" + workerID + "' AND Timestamp = '"
                        + timestamp + "';";
        return queryAndGet(sql, 1);
    }

    @Override
    public List<Persistable> getLatestUpdateAllWorkers(String sessionID, String typeID) {
        String sql = "SELECT workerId, MAX(Timestamp) FROM " + TABLE_NAME_UPDATES + " WHERE SessionID ='"
                + sessionID + "' AND " + "TypeID = '" + typeID + "' GROUP BY workerId";

        Map<String,Long> m = new HashMap<>();
        try (Statement statement = connection.createStatement()) {
            ResultSet rs = statement.executeQuery(sql);
            while (rs.next()) {
                String wid = rs.getString(1);
                long ts = rs.getLong(2);
                m.put(wid, ts);
            }
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }

        List<Persistable> out = new ArrayList<>();
        for(String s : m.keySet()){
            out.add(getUpdate(sessionID, typeID, s, m.get(s)));
        }
        return out;
    }

    @Override
    public List<Persistable> getAllUpdatesAfter(String sessionID, String typeID, String workerID, long timestamp) {
        String sql = "SELECT * FROM " + TABLE_NAME_UPDATES + " WHERE SessionID = '" + sessionID + "' AND TypeID = '"
                        + typeID + "' AND workerId = '" + workerID + "' AND Timestamp > " + timestamp + ";";
        try (Statement statement = connection.createStatement()) {
            ResultSet rs = statement.executeQuery(sql);
            List<Persistable> out = new ArrayList<>();
            while (rs.next()) {
                byte[] bytes = rs.getBytes(6);
                out.add((Persistable) deserialize(bytes));
            }
            return out;
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public List<Persistable> getAllUpdatesAfter(String sessionID, String typeID, long timestamp) {
        String sql = "SELECT ObjectBytes FROM " + TABLE_NAME_UPDATES + " WHERE SessionID = '" + sessionID + "'  "
                        + "AND TypeID = '" + typeID + "' AND Timestamp > " + timestamp + ";";
        return queryUpdates(sql);
    }

    @Override
    public long[] getAllUpdateTimes(String sessionID, String typeID, String workerID) {
        String sql = "SELECT Timestamp FROM " + TABLE_NAME_UPDATES + " WHERE SessionID = '" + sessionID + "'  "
                + "AND TypeID = '" + typeID + "' AND workerID = '" + workerID + "';";
        try (Statement statement = connection.createStatement()) {
            ResultSet rs = statement.executeQuery(sql);
            LongArrayList list = new LongArrayList();
            while (rs.next()) {
                list.add(rs.getLong(1));
            }
            return list.toLongArray();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public List<Persistable> getUpdates(String sessionID, String typeID, String workerID, long[] timestamps) {
        if(timestamps == null || timestamps.length == 0){
            return Collections.emptyList();
        }

        StringBuilder sb = new StringBuilder();
        sb.append("SELECT ObjectBytes FROM ").append(TABLE_NAME_UPDATES).append(" WHERE SessionID = '").append(sessionID)
                .append("' AND TypeID = '").append(typeID).append("' AND workerID='").append(workerID)
                .append("'  AND Timestamp IN (");

        for( int i=0; i<timestamps.length; i++ ){
            if(i > 0){
                sb.append(",");
            }
            sb.append(timestamps[i]);
        }
        sb.append(");");

        String sql = sb.toString();
        return queryUpdates(sql);
    }

    private List<Persistable> queryUpdates(String sql){
        try (Statement statement = connection.createStatement()) {
            ResultSet rs = statement.executeQuery(sql);
            List<Persistable> out = new ArrayList<>();
            while (rs.next()) {
                byte[] bytes = rs.getBytes(1);
                out.add((Persistable) deserialize(bytes));
            }
            return out;
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public StorageMetaData getStorageMetaData(String sessionID, String typeID) {
        String sql = "SELECT ObjectBytes FROM " + TABLE_NAME_METADATA + " WHERE SessionID = '" + sessionID
                        + "' AND TypeID = '" + typeID + "' LIMIT 1;";
        return queryAndGet(sql, 1);
    }

    @Override
    public void registerStatsStorageListener(StatsStorageListener listener) {
        listeners.add(listener);
    }

    @Override
    public void deregisterStatsStorageListener(StatsStorageListener listener) {
        listeners.remove(listener);
    }

    @Override
    public void removeAllListeners() {
        listeners.clear();
    }

    @Override
    public List<StatsStorageListener> getListeners() {
        return new ArrayList<>(listeners);
    }

    @Override
    public String toString() {
        return "J7FileStatsStorage(file=" + file + ")";
    }

    protected void notifyListeners(List<StatsStorageEvent> sses) {
        if (sses == null || sses.isEmpty() || listeners.isEmpty())
            return;
        for (StatsStorageListener l : listeners) {
            for (StatsStorageEvent e : sses) {
                l.notify(e);
            }
        }
    }
}
