package org.tribbloid.ispark.display

import java.net.URL

import org.apache.spark.sql.SchemaRDD
import org.json4s.DefaultFormats
import org.pegdown.{Extensions, PegDownProcessor}

object Display {

  case class Math(math: String) extends LatexDisplayObject {
    override val toLatex = "$$" + math + "$$"
  }

  case class Latex(latex: String) extends LatexDisplayObject {
    override val toLatex = latex
  }

  case class HTML(code: String) extends HTMLDisplayObject {
    override val toHTML: String = code
  }

  object MarkdownProcessor extends PegDownProcessor(Extensions.ALL)

  case class Markdown(code: String) extends HTMLDisplayObject {
    override val toHTML: String = {
      MarkdownProcessor.markdownToHtml(code)
    }
  }

  case class Table(
                    rdd: SchemaRDD,
                    limit:Int = 1000
                    ) extends HTMLDisplayObject {
    assert(limit<=1000) //or parsing timeout

    override val toHTML: String = {
      val schema = rdd.schema.fieldNames
      val header = schema.mkString(" | ")
      val splitter = schema.map(v => "-").mkString(" | ")
      rdd.persist()
      val size = rdd.count()
      val rows = rdd.take(limit)
      rdd.unpersist()
      val info =
        if (size < limit) s"##### returned $size rows in total:"
      else s"##### returned $size rows in total but only $limit of them are displayed:"
      val body = rows.map(row => row.mkString(" | "))
      val all = Seq(info, header, splitter) ++ body
      val code = all.mkString("\n")

      MarkdownProcessor.markdownToHtml(code)
    }
  }

  class IFrame(src: URL, width: Int, height: Int) extends HTMLDisplayObject {
    protected def iframe() =
      <iframe width={width.toString}
              height={height.toString}
              src={src.toString}
              frameborder="0"
              allowfullscreen="allowfullscreen"></iframe>

    override val toHTML = iframe().toString()
  }

  object IFrame {
    def apply(src: URL, width: Int, height: Int): IFrame = new IFrame(src, width, height)

    def apply(src: String, width: Int, height: Int): IFrame = new IFrame(new URL(src), width, height)
  }

  case class YouTubeVideo(id: String, width: Int = 400, height: Int = 300)
    extends IFrame(new URL("https", "www.youtube.com", s"/embed/$id"), width, height)

  case class VimeoVideo(id: String, width: Int = 400, height: Int = 300)
    extends IFrame(new URL("https", "player.vimeo.com", s"/video/$id"), width, height)

  case class ScribdDocument(id: String, width: Int = 400, height: Int = 300)
    extends IFrame(new URL("https", "www.scribd.com", s"/embeds/$id/content"), width, height)

  case class ImageURL(url: URL, width: Option[Int], height: Option[Int]) extends HTMLDisplayObject {
    override val toHTML = <img src={url.toString}
                      width={width.map(w => xml.Text(w.toString))}
                      height={height.map(h => xml.Text(h.toString))}></img> toString()
  }

  object ImageURL {
    def apply(url: URL): ImageURL = ImageURL(url, None, None)

    def apply(url: String): ImageURL = ImageURL(new URL(url))

    def apply(url: URL, width: Int, height: Int): ImageURL = ImageURL(url, Some(width), Some(height))

    def apply(url: String, width: Int, height: Int): ImageURL = ImageURL(new URL(url), width, height)
  }

  //disabled because Json display is only supported in extension
//  case class Json[T <: AnyRef](obj: T) extends JSONDisplayObject {
//
//    implicit val formats = DefaultFormats
//    import org.json4s.jackson.Serialization
//
//    override val toJSON: String = {
//      Serialization.write(obj) //TODO: Cannot serialize class created in interpreter
//    }
//  }
}