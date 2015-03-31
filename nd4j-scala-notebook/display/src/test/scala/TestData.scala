import org.scalatest.FunSuite
import org.tribbloid.ispark.display.Display.Markdown
import org.tribbloid.ispark.display.{MIME, Data}

/**
 * Created by peng on 1/6/15.
 */
class TestData extends FunSuite {

  test("plain string") {
    val text = "I'm a string"

    val data = Data.parse(text)
    println(data)
    assert(data.items.map(_._1).contains(MIME.`text/plain`))
    assert(data.items.size === 1)
  }

  test("HTML") {
    val html =
      <table>
        <tr>
          <th>Header 1</th>
          <th>Header 2</th>
        </tr>
        <tr>
          <td>row 1, cell 1</td>
          <td>row 1, cell 2</td>
        </tr>
        <tr>
          <td>row 2, cell 1</td>
          <td>row 2, cell 2</td>
        </tr>
      </table>

    val data = Data.parse(html)
    println(data)
    assert(data.items.map(_._1).contains(MIME.`text/plain`))
    assert(data.items.map(_._1).contains(MIME.`text/html`))
    assert(data.items.size === 2)
  }

  test("Markdown") {
    val md = Markdown(
      """
        |### title
      """.stripMargin)

    val data = Data.parse(md)
    println(data)
    assert(data.items.map(_._1).contains(MIME.`text/plain`))
    assert(data.items.map(_._1).contains(MIME.`text/html`))
    assert(data.items.size === 2)
  }
}