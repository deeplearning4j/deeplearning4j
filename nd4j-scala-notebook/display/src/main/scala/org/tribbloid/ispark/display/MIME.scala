package org.tribbloid.ispark.display

import java.awt.image.RenderedImage
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO

import com.sun.org.apache.xml.internal.security.utils.Base64

import scala.xml.NodeSeq

sealed abstract class MIME(val name: String) {
  type Parse = PartialFunction[Any, Option[String]]

  def parse: Parse
}

object MIME {
  val all: List[MIME] = List(
    `text/plain`,
    `text/html`,
//    `text/markdown`, //not supported at the moment
    `text/latex`,
    `application/json`,
    `application/javascript`,
    `image/png`,
    `image/jpeg`,
    `image/svg+xml`
  )

  case object `text/plain` extends MIME("text/plain") {
    override def parse = {
      case obj: Any => Some(scala.runtime.ScalaRunTime.stringOf(obj))
      case _ => None
    }
  }

  case object `text/html` extends MIME("text/html") {
    override def parse = {
      case obj: NodeSeq => Some(obj.toString())
      case obj: HTMLDisplayObject => Some(obj.toHTML)
      case _ => None
    }
  }

//  case object `text/markdown` extends MIME("text/markdown") {
//    override def parse = {
//      case obj: MarkdownDisplayObject => Some(obj.toMarkdown)
//      case _ => None
//    }
//  }

  case object `text/latex` extends MIME("text/latex") {
    override def parse = {
      case obj: LatexDisplayObject => Some(obj.toLatex)
      case _ => None
    }
  }

  case object `application/json` extends MIME("application/json") {
    override def parse = {
      case obj: JSONDisplayObject => Some(obj.toJSON)
      case _ => None
    }
  }

  case object `application/javascript` extends MIME("application/javascript") {
    override def parse = {
      case obj: JavascriptDisplayObject => Some(obj.toJavascript)
      case _ => None
    }
  }

  case object `image/png` extends MIME("image/png") {

    private def encodeImage(format: String)(image: RenderedImage): String = {
      val output = new ByteArrayOutputStream()
      val bytes = try {
        ImageIO.write(image, format, output)
        output.toByteArray
      } finally {
        output.close()
      }
      Base64.encode(bytes)
    }

    override def parse = {
      case obj: RenderedImage => Some(encodeImage("PNG")(obj))
      case obj: PNGDisplayObject => Some(obj.toPNG)
      case _ => None
    }
  }

  case object `image/jpeg` extends MIME("image/jpeg") {
    override def parse = {
      case obj: JPEGDisplayObject => Some(obj.toJPEG)
      case _ => None
    }
  }

  case object `image/svg+xml` extends MIME("image/svg+xml") {
    override def parse = {
      case obj: SVGDisplayObject => Some(obj.toSVG)
      case _ => None
    }
  }
}
