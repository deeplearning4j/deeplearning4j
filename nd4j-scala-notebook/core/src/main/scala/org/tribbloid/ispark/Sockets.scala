package org.tribbloid.ispark

import org.zeromq.ZMQ

class Sockets(profile: Profile) {
    val ctx = ZMQ.context(1)

    val publish = ctx.socket(ZMQ.PUB)
    val requests = ctx.socket(ZMQ.ROUTER)
    val control = ctx.socket(ZMQ.ROUTER)
    val stdin = ctx.socket(ZMQ.ROUTER)
    val heartbeat = ctx.socket(ZMQ.REP)

    private def toURI(port: Int) =
        s"${profile.transport}://${profile.ip}:$port"

    publish.bind(toURI(profile.iopub_port))
    requests.bind(toURI(profile.shell_port))
    control.bind(toURI(profile.control_port))
    stdin.bind(toURI(profile.stdin_port))
    heartbeat.bind(toURI(profile.hb_port))

    def terminate() {
        publish.close()
        requests.close()
        control.close()
        stdin.close()
        heartbeat.close()

        ctx.term()
    }
}
