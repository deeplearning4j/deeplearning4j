
package org.deeplearning4j.ui.views.html.histogram

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Histogram_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Histogram extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.1*/("""<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8"/>
        <title>Weights/Gradients/Score</title>
        <style>
        .bar rect """),format.raw/*7.19*/("""{"""),format.raw/*7.20*/("""
            """),format.raw/*8.13*/("""fill: steelblue;
            shape-rendering: crispEdges;
        """),format.raw/*10.9*/("""}"""),format.raw/*10.10*/("""

        """),format.raw/*12.9*/(""".bar text """),format.raw/*12.19*/("""{"""),format.raw/*12.20*/("""
            """),format.raw/*13.13*/("""fill: #EFEFEF;
        """),format.raw/*14.9*/("""}"""),format.raw/*14.10*/("""

        """),format.raw/*16.9*/(""".area """),format.raw/*16.15*/("""{"""),format.raw/*16.16*/("""
            """),format.raw/*17.13*/("""fill: steelblue;
        """),format.raw/*18.9*/("""}"""),format.raw/*18.10*/("""

        """),format.raw/*20.9*/(""".axis path, .axis line """),format.raw/*20.32*/("""{"""),format.raw/*20.33*/("""
            """),format.raw/*21.13*/("""fill: none;
            stroke: #000;
            shape-rendering: crispEdges;
        """),format.raw/*24.9*/("""}"""),format.raw/*24.10*/("""

        """),format.raw/*26.9*/(""".tick line """),format.raw/*26.20*/("""{"""),format.raw/*26.21*/("""
            """),format.raw/*27.13*/("""opacity: 0.2;
            shape-rendering: crispEdges;
        """),format.raw/*29.9*/("""}"""),format.raw/*29.10*/("""

        """),format.raw/*31.9*/("""path """),format.raw/*31.14*/("""{"""),format.raw/*31.15*/("""
            """),format.raw/*32.13*/("""stroke: steelblue;
            stroke-width: 2;
            fill: none;
        """),format.raw/*35.9*/("""}"""),format.raw/*35.10*/("""

        """),format.raw/*37.9*/(""".legend """),format.raw/*37.17*/("""{"""),format.raw/*37.18*/("""
            """),format.raw/*38.13*/("""font-size: 12px;
            text-anchor: middle;
        """),format.raw/*40.9*/("""}"""),format.raw/*40.10*/("""

        """),format.raw/*42.9*/(""".brush .extent """),format.raw/*42.24*/("""{"""),format.raw/*42.25*/("""
            """),format.raw/*43.13*/("""stroke: #fff;
            fill-opacity: .125;
            shape-rendering: crispEdges;
        """),format.raw/*46.9*/("""}"""),format.raw/*46.10*/("""

        """),format.raw/*48.9*/("""</style>

            <!-- jQuery -->
        <script src="/assets/legacy/jquery-2.2.0.min.js"></script>

        <link href='/assets/legacy/roboto.css' rel='stylesheet' type='text/css'>

            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap.min.css" />

            <!-- Optional theme -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-theme.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/legacy/bootstrap.min.js" ></script>

            <!-- d3 -->
        <script src="/assets/legacy/d3.v3.min.js" charset="utf-8"></script>

        <script src="/assets/legacy/jquery-fileupload.js"></script>

            <!-- Booststrap Notify plugin-->
        <script src="/assets/legacy/bootstrap-notify.min.js"></script>

            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-select.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/legacy/bootstrap-select.min.js"></script>

            <!-- DateTime formatter-->
        <script src="/assets/legacy/DateTimeFormat.js"></script>

        <script src="/assets/legacy/renderWeightsProper.js"></script>

        <script src="/assets/legacy/common.js"></script>

        <style>
        body """),format.raw/*86.14*/("""{"""),format.raw/*86.15*/("""
            """),format.raw/*87.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*91.9*/("""}"""),format.raw/*91.10*/("""

        """),format.raw/*93.9*/(""".hd """),format.raw/*93.13*/("""{"""),format.raw/*93.14*/("""
            """),format.raw/*94.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*97.9*/("""}"""),format.raw/*97.10*/("""

        """),format.raw/*99.9*/(""".block """),format.raw/*99.16*/("""{"""),format.raw/*99.17*/("""
            """),format.raw/*100.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*105.9*/("""}"""),format.raw/*105.10*/("""

        """),format.raw/*107.9*/(""".hd-small """),format.raw/*107.19*/("""{"""),format.raw/*107.20*/("""
            """),format.raw/*108.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*111.9*/("""}"""),format.raw/*111.10*/("""

        """),format.raw/*113.9*/(""".charts, .chart """),format.raw/*113.25*/("""{"""),format.raw/*113.26*/("""
            """),format.raw/*114.13*/("""font-size: 10px;
            font-color: #000000;
            position: relative;
        """),format.raw/*117.9*/("""}"""),format.raw/*117.10*/("""

        """),format.raw/*119.9*/(""".scoreboard """),format.raw/*119.21*/("""{"""),format.raw/*119.22*/("""
            """),format.raw/*120.13*/("""position: absolute;
            top: 20px;
            right: 10px;
            z-index: 1000;
            font-size: 11px;
        """),format.raw/*125.9*/("""}"""),format.raw/*125.10*/("""

        """),format.raw/*127.9*/(""".score """),format.raw/*127.16*/("""{"""),format.raw/*127.17*/("""
            """),format.raw/*128.13*/("""font-size: 11px;
        """),format.raw/*129.9*/("""}"""),format.raw/*129.10*/("""
        """),format.raw/*130.9*/("""</style>

    </head>
    <body>
        <table style="width: 100%;
            padding: 5px;" class="hd">
            <tbody>
                <tr>
                    <td style="width: 48px;"><a href="/"><img src="/assets/legacy/deeplearning4j.img" border="0"/></a></td>
                    <td>DeepLearning4j UI</td>
                    <td style="width: 512px;
                        text-align: right;" class="hd-small">&nbsp; Available sessions: <select class="selectpicker" id="sessionSelector" onchange="window.location.href = 'weights?sid=' + this.options[this.selectedIndex].value ;" style="color: #000000;
                        display: inline-block;
                        width: 256px;">
                        <option value="0" selected="selected">Pick a session to track</option>
                    </select> &nbsp;&nbsp;
                        <script>
            buildSessionSelector("HISTOGRAM");
          </script>
                    </td>
                    <td style="width: 256px;" class="hd-small">&nbsp;Updated at: <b><span id="updatetime">No updates so far</span></b> &nbsp;</td>
                </tr>
            </tbody>
        </table>

        <div style="width: 100%;
            text-align: center;">
            <div id="display" style="width: 1540px;
                height: 900px;
                text-align: left;
                background-color: #FFFFFF;
                display: inline-block;
                overflow: hidden; ">
                <div id="scores" style="background-color: #EFEFEF;
                    display: block;
                    float: left;
                    width: 750px;
                    height: 400px;
                    border: 1px solid #CECECE;
                    margin: 10px;">
                    <h5>&nbsp;&nbsp;Score vs. Iteration #</h5>
                    <div class="chart" id="schart">
        </div>
                </div>
                <div id="model" style="position: relative;
                    background-color: #EFEFEF;
                    display: block;
                    float: left;
                    width: 750px;
                    height: 400px;
                    border: 1px solid #CECECE;
                    margin: 10px;">
                    <h5>&nbsp;&nbsp;Model</h5>
                    <div class="charts"></div>
                    <div style="position: absolute;
                        top: 5px;
                        right: 5px;">
                        <select id="modelSelector" onchange="selectModel();">
          </select>
                    </div>
                </div>
                <div id="gradient" style="position: relative;
                    background-color: #EFEFEF;
                    display: block;
                    float: left;
                    width: 750px;
                    height: 400px;
                    border: 1px solid #CECECE;
                    margin: 10px;">
                    <h5>&nbsp;&nbsp;Gradient</h5>
                    <div class="charts"></div>
                    <div style="position: absolute;
                        top: 5px;
                        right: 5px;">
                        <select id="gradientSelector" onchange="selectGradient();">
          </select>
                    </div>
                </div>
                <div id="magnitudes" style="position: relative;
                    background-color: #EFEFEF;
                    display: block;
                    float: left;
                    width: 750px;
                    height: 400px;
                    border: 1px solid #CECECE;
                    margin: 10px;">
                    <h5>&nbsp;&nbsp;Mean Magnitudes: Parameters and Updates</h5>
                    <div class="charts"></div>
                    <div style="position: absolute;
                        top: 5px;
                        right: 5px;">
                        <select id="magnitudeSelector" onchange="selectMagnitude();">
          </select>
                    </div>
                </div>
                    <!--<div id="lastupdate">
            <div class="updatetime">-1</div>
        </div>-->
            </div>

                <!--
    <div style="display: block;">
        nav bar
    </div> -->
        </div>
            <!--
<div id="score" style="display: inline-block; width: 650px; height: 400px; border: 1px solid #CECECE;">
    <h4>Score</h4>
    <div class="score"></div>
</div>-->

    </body>
</html>"""))
      }
    }
  }

  def render(): play.twirl.api.HtmlFormat.Appendable = apply()

  def f:(() => play.twirl.api.HtmlFormat.Appendable) = () => apply()

  def ref: this.type = this

}


}

/**/
object Histogram extends Histogram_Scope0.Histogram
              /*
                  -- GENERATED --
                  DATE: Tue Oct 25 15:17:44 PDT 2016
                  SOURCE: /Users/ejunprung/skymind-ui/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/histogram/Histogram.scala.html
                  HASH: dabb2cbac56bd1753a2b8ed3d82bf89534bd4463
                  MATRIX: 649->0|823->147|851->148|891->161|984->227|1013->228|1050->238|1088->248|1117->249|1158->262|1208->285|1237->286|1274->296|1308->302|1337->303|1378->316|1430->341|1459->342|1496->352|1547->375|1576->376|1617->389|1731->476|1760->477|1797->487|1836->498|1865->499|1906->512|1996->575|2025->576|2062->586|2095->591|2124->592|2165->605|2272->685|2301->686|2338->696|2374->704|2403->705|2444->718|2529->776|2558->777|2595->787|2638->802|2667->803|2708->816|2830->911|2859->912|2896->922|4304->2302|4333->2303|4374->2316|4528->2443|4557->2444|4594->2454|4626->2458|4655->2459|4696->2472|4815->2564|4844->2565|4881->2575|4916->2582|4945->2583|4987->2596|5170->2751|5200->2752|5238->2762|5277->2772|5307->2773|5349->2786|5469->2878|5499->2879|5537->2889|5582->2905|5612->2906|5654->2919|5772->3009|5802->3010|5840->3020|5881->3032|5911->3033|5953->3046|6113->3178|6143->3179|6181->3189|6217->3196|6247->3197|6289->3210|6342->3235|6372->3236|6409->3245
                  LINES: 25->1|31->7|31->7|32->8|34->10|34->10|36->12|36->12|36->12|37->13|38->14|38->14|40->16|40->16|40->16|41->17|42->18|42->18|44->20|44->20|44->20|45->21|48->24|48->24|50->26|50->26|50->26|51->27|53->29|53->29|55->31|55->31|55->31|56->32|59->35|59->35|61->37|61->37|61->37|62->38|64->40|64->40|66->42|66->42|66->42|67->43|70->46|70->46|72->48|110->86|110->86|111->87|115->91|115->91|117->93|117->93|117->93|118->94|121->97|121->97|123->99|123->99|123->99|124->100|129->105|129->105|131->107|131->107|131->107|132->108|135->111|135->111|137->113|137->113|137->113|138->114|141->117|141->117|143->119|143->119|143->119|144->120|149->125|149->125|151->127|151->127|151->127|152->128|153->129|153->129|154->130
                  -- GENERATED --
              */
          