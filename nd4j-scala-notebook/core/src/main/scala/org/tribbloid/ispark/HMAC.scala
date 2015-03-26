package org.tribbloid.ispark

import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

object HMAC {
    def apply(key: String, algorithm: Option[String]=None): HMAC =
        if (key.isEmpty) NoHMAC else new DoHMAC(key)
}

sealed trait HMAC {
    def hexdigest(args: Seq[String]): String

    final def apply(args: String*) = hexdigest(args)
}

final class DoHMAC(key: String, algorithm: Option[String]=None) extends HMAC {
    private val _algorithm = algorithm getOrElse "hmac-sha256" replace ("-", "")
    private val mac = Mac.getInstance(_algorithm)
    private val keySpec = new SecretKeySpec(key.getBytes, _algorithm)
    mac.init(keySpec)

    def hexdigest(args: Seq[String]): String = {
        mac synchronized {
            args.map(_.getBytes).foreach(mac.update)
            Util.hex(mac.doFinal())
        }
    }
}

object NoHMAC extends HMAC {
    def hexdigest(args: Seq[String]): String = ""
}
