package org.tribbloid.ispark

class UUID private (uuid: java.util.UUID, dashes: Boolean=true, upper: Boolean=false) {
    override def toString: String = {
        val repr0 = uuid.toString
        val repr1 = if (dashes) repr0 else repr0.replace("-", "")
        val repr2 = if (upper) repr1.toUpperCase else repr1
        repr2
    }
}

object UUID {
    def fromString(uuid: String): Option[UUID] = {
        val (actualUuid, dashes) =
            if (uuid.contains("-")) (uuid, true)
            else (List(uuid.slice( 0,  8),
                       uuid.slice( 8, 12),
                       uuid.slice(12, 16),
                       uuid.slice(16, 20),
                       uuid.slice(20, 32)).mkString("-"), false)

        val upper = uuid.exists("ABCDF" contains _)

        try {
            Some(new UUID(java.util.UUID.fromString(actualUuid), dashes, upper))
        } catch {
            case e: java.lang.IllegalArgumentException =>
                None
        }
    }

    def uuid4(): UUID = new UUID(java.util.UUID.randomUUID)
}
