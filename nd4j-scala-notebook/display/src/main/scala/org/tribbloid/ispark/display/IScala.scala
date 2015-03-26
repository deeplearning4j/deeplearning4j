package org.tribbloid.ispark.display

trait Conn {
  def display_data(data: Data): Unit
}

object IScala {
  def withConn[T](conn: Conn)(block: => T): T = {
    _conn = Some(conn)
    try {
      block
    } finally {
      _conn = None
    }
  }

  private var _conn: Option[Conn] = None
  private def conn: Conn = _conn getOrElse sys.error("Not in IScala")

  def display_data(data: Data) {
    conn.display_data(data)
  }
}
