package org.tribbloid.ispark.db

import java.sql.Timestamp

import scala.slick.driver.SQLiteDriver.simple._
import Database.dynamicSession
import scalax.file.Path

class Sessions(tag: Tag) extends Table[(Int, Timestamp, Option[Timestamp], Option[Int], String)](tag, "sessions") {
  def session = column[Int]("session", O.PrimaryKey, O.AutoInc)
  def start = column[Timestamp]("start")
  def end = column[Option[Timestamp]]("end")
  def num_cmds = column[Option[Int]]("num_cmds")
  def remark = column[String]("remark")

  def * = (session, start, end, num_cmds, remark)
}

class History(tag: Tag) extends Table[(Int, Int, String, String)](tag, "history") {
  def session = column[Int]("session")
  def line = column[Int]("line")
  def source = column[String]("source")
  def source_raw = column[String]("source_raw")

  def pk = primaryKey("pk_session_line", (session, line))

  def * = (session, line, source, source_raw)
}

class OutputHistory(tag: Tag) extends Table[(Int, Int, String)](tag, "output_history") {
  def session = column[Int]("session")
  def line = column[Int]("line")
  def output = column[String]("output")

  def pk = primaryKey("pk_session_line", (session, line))

  def * = (session, line, output)
}

object DB {
  val Sessions = TableQuery[Sessions]
  val History = TableQuery[History]
  val OutputHistory = TableQuery[OutputHistory]

  private lazy val dbPath = { //TODO: backport! read from parameters
    val home = Path.fromString(System.getProperty("user.home"))
    val profile = home / ".config" / "ipython" / "profile_scala"
    if (!profile.exists) profile.createDirectory()
    profile / "history.sqlite" path
  }

  lazy val db = {
    val db = Database.forURL(s"jdbc:sqlite:$dbPath", driver="org.sqlite.JDBC")
    db.withDynSession {
      Seq(Sessions, History, OutputHistory) foreach { table =>
        try {
          table.ddl.create
        } catch {
          case error: java.sql.SQLException if error.getMessage contains "already exists" =>
        }
      }
    }
    db
  }

  import db.withDynSession

  private def now = new Timestamp(System.currentTimeMillis)

  def newSession(): Int = withDynSession {
    (Sessions returning Sessions.map(_.session)) += (0, now, None, None, "")
  }

  def endSession(session: Int)(num_cmds: Int): Unit = withDynSession {
    val q = for { s <- Sessions if s.session === session } yield (s.end, s.num_cmds)
    q.update(Some(now), Some(num_cmds))
  }

  def addHistory(session: Int)(line: Int, source: String): Unit = withDynSession {
    History += (session, line, source, source)
  }

  def addOutputHistory(session: Int)(line: Int, output: String): Unit = withDynSession {
    OutputHistory += (session, line, output)
  }
}
