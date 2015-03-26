package org.tribbloid.ispark

import org.tribbloid.ispark.json.Json

case class Profile(
    ip: String,
    transport: String,
    stdin_port: Int,
    control_port: Int,
    hb_port: Int,
    shell_port: Int,
    iopub_port: Int,
    key: String,
    signature_scheme: Option[String])

object Profile {
    implicit val ProfileJSON = Json.format[Profile]

    lazy val default = {
        val port0 = 5678
        Profile(ip="127.0.0.1",
                transport="tcp",
                stdin_port=port0,
                control_port=port0+1,
                hb_port=port0+2,
                shell_port=port0+3,
                iopub_port=port0+4,
                key=UUID.uuid4().toString,
                signature_scheme=None)
    }
}
