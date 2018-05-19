
package org.deeplearning4j.ui.views.html.flow

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Flow_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Flow extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.1*/("""<html>
    <head>
        <meta charset="utf-8" />

        <title>Flow overview</title>


            <!-- jQuery -->
        <script src="/assets/legacy/jquery-2.2.0.min.js"></script>

        <link href='/assets/legacy/roboto.css' rel='stylesheet' type='text/css'>

            <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap.min.css" />

            <!-- Optional theme -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-theme.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        <script src="/assets/legacy/bootstrap.min.js" ></script>


            <!-- Booststrap Notify plugin-->
        <script src="/assets/legacy/bootstrap-notify.min.js"></script>

            <!-- DateTime formatter-->
        <script src="/assets/legacy/DateTimeFormat.js"></script>

            <!-- d3 -->
        <script src="/assets/legacy/d3.v3.min.js" charset="utf-8"></script>

        <script src="/assets/legacy/Connection.js"></script>
        <script src="/assets/legacy/Layer.js"></script>
        <script src="/assets/legacy/Layers.js"></script>

        <script src="/assets/legacy/common.js"></script>

        <script src="/assets/legacy/renderFlow.js"></script>
        <style>
        body """),format.raw/*40.14*/("""{"""),format.raw/*40.15*/("""
            """),format.raw/*41.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*45.9*/("""}"""),format.raw/*45.10*/("""
        """),format.raw/*46.9*/(""".hd """),format.raw/*46.13*/("""{"""),format.raw/*46.14*/("""
            """),format.raw/*47.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*50.9*/("""}"""),format.raw/*50.10*/("""
        """),format.raw/*51.9*/(""".block """),format.raw/*51.16*/("""{"""),format.raw/*51.17*/("""
            """),format.raw/*52.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*57.9*/("""}"""),format.raw/*57.10*/("""
        """),format.raw/*58.9*/(""".hd-small """),format.raw/*58.19*/("""{"""),format.raw/*58.20*/("""
            """),format.raw/*59.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*62.9*/("""}"""),format.raw/*62.10*/("""
        """),format.raw/*63.9*/(""".charts, .chart """),format.raw/*63.25*/("""{"""),format.raw/*63.26*/("""
            """),format.raw/*64.13*/("""font-size: 10px;
            font-color: #000000;
        """),format.raw/*66.9*/("""}"""),format.raw/*66.10*/("""
        """),format.raw/*67.9*/(""".tooltip """),format.raw/*67.18*/("""{"""),format.raw/*67.19*/("""
            """),format.raw/*68.13*/("""position: absolute;
            top: 140px;
            left: 0px;
            right: 0;
            width: 220px;
            padding: 2px 0;
            background-color: #000;
            background-color: rgba(0, 0, 0, 0.65);
            color: #fff;
            opacity: 0;
            transition: opacity .5s ease-in-out;
            text-align: center;
            font-family: Arial;
            font-size: 14px;
            z-index: 100;
        """),format.raw/*83.9*/("""}"""),format.raw/*83.10*/("""
        """),format.raw/*84.9*/(""".viewpanel """),format.raw/*84.20*/("""{"""),format.raw/*84.21*/("""
            """),format.raw/*85.13*/("""position: absolute;
            background-color: #FFF;
            top: 60px;
            bottom: 0px;
        """),format.raw/*89.9*/("""}"""),format.raw/*89.10*/("""

        """),format.raw/*91.9*/(""".perftd """),format.raw/*91.17*/("""{"""),format.raw/*91.18*/("""
            """),format.raw/*92.13*/("""padding-right: 10px;
            padding-bottom: 1px;
            font-family: Arial;
            font-size: 14px;
        """),format.raw/*96.9*/("""}"""),format.raw/*96.10*/("""

        """),format.raw/*98.9*/(""".bar rect """),format.raw/*98.19*/("""{"""),format.raw/*98.20*/("""
            """),format.raw/*99.13*/("""fill: steelblue;
            shape-rendering: crispEdges;
        """),format.raw/*101.9*/("""}"""),format.raw/*101.10*/("""

        """),format.raw/*103.9*/(""".bar text """),format.raw/*103.19*/("""{"""),format.raw/*103.20*/("""
            """),format.raw/*104.13*/("""fill: #EFEFEF;
        """),format.raw/*105.9*/("""}"""),format.raw/*105.10*/("""

        """),format.raw/*107.9*/(""".area """),format.raw/*107.15*/("""{"""),format.raw/*107.16*/("""
            """),format.raw/*108.13*/("""fill: steelblue;
        """),format.raw/*109.9*/("""}"""),format.raw/*109.10*/("""

        """),format.raw/*111.9*/(""".axis path, .axis line """),format.raw/*111.32*/("""{"""),format.raw/*111.33*/("""
            """),format.raw/*112.13*/("""fill: none;
            stroke: #000;
            stroke-width: 1.5;
            shape-rendering: crispEdges;
        """),format.raw/*116.9*/("""}"""),format.raw/*116.10*/("""

        """),format.raw/*118.9*/(""".tick line """),format.raw/*118.20*/("""{"""),format.raw/*118.21*/("""
            """),format.raw/*119.13*/("""opacity: 0.2;
            stroke-width: 1.5;
            shape-rendering: crispEdges;
        """),format.raw/*122.9*/("""}"""),format.raw/*122.10*/("""

        """),format.raw/*124.9*/(""".tick """),format.raw/*124.15*/("""{"""),format.raw/*124.16*/("""
            """),format.raw/*125.13*/("""font-size: 9px;
        """),format.raw/*126.9*/("""}"""),format.raw/*126.10*/("""

        """),format.raw/*128.9*/("""path """),format.raw/*128.14*/("""{"""),format.raw/*128.15*/("""
            """),format.raw/*129.13*/("""stroke: steelblue;
            stroke-width: 1.5;
            fill: none;
        """),format.raw/*132.9*/("""}"""),format.raw/*132.10*/("""

        """),format.raw/*134.9*/(""".legend """),format.raw/*134.17*/("""{"""),format.raw/*134.18*/("""
            """),format.raw/*135.13*/("""font-size: 12px;
            text-anchor: middle;
        """),format.raw/*137.9*/("""}"""),format.raw/*137.10*/("""

        """),format.raw/*139.9*/(""".layerDesc """),format.raw/*139.20*/("""{"""),format.raw/*139.21*/("""
            """),format.raw/*140.13*/("""font-family: Arial;
            font-size: 12px;
        """),format.raw/*142.9*/("""}"""),format.raw/*142.10*/("""

        """),format.raw/*144.9*/(""".brush .extent """),format.raw/*144.24*/("""{"""),format.raw/*144.25*/("""
            """),format.raw/*145.13*/("""stroke: #fff;
            stroke-width: 1.5;
            fill-opacity: .125;
            shape-rendering: crispEdges;
        """),format.raw/*149.9*/("""}"""),format.raw/*149.10*/("""
        """),format.raw/*150.9*/("""</style>
    </head>
    <body>
        <table style="width: 100%; padding: 5px;" class="hd">
            <tbody>
                <tr>
                    <td style="width: 48px;"><a href="/"><img src="/assets/legacy/deeplearning4j.img" border="0"/></a></td>
                    <td>DeepLearning4j UI</td>
                    <td style="width: 512px; text-align: right;" class="hd-small">&nbsp; Available sessions: <select class="selectpicker" id="sessionSelector" onchange="window.location.href = 'flow?sid='+ this.options[this.selectedIndex].value ;" style="color: #000000; display: inline-block; width: 256px;">
                        <option value="0" selected="selected">Pick a session to track</option>
                    </select>&nbsp;&nbsp;
                        <script>
                            buildSessionSelector2("flow/listSessions","");
                        </script>
                    </td>
                    <td style="width: 256px;" class="hd-small">&nbsp;Updated at: <b><span id="updatetime">No updates so far</span></b>&nbsp;</td>
                </tr>
            </tbody>
        </table>
        <br /> <br />
        <div style="width: 100%; text-align: center;">
            <div id="display" style="display: inline-block; width: 900px;">
                    <!-- NN rendering pane -->
            </div>
        </div>
        <div id="tooltip" class="tooltip">
                &nbsp;
        </div>

            <!-- Left view panel -->
        <div style="left: 10px; width: 400px;" class="viewpanel">
            <center>
                <table style="margin: 10px; width: 200px;">
                    <tr>
                        <td><b>Score vs iteration:</b></td>
                    </tr>
                </table>
            </center>
            <div id="scoreChart" style="background-color: #FFF; height: 250px;">
                    &nbsp;
            </div>
            <br/>
            <div style="width: 100%; background-color: #FFF; text-align:center; display: block; ">
                <center>
                    <table style="margin: 10px; width: 200px;">
                        <thead style="width: 200px;">
                            <td colspan="2"><b>Model training status:</b></td>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="perftd">Current score:</td>
                                <td class="perftd" id="ss">0.0</td>
                            </tr>
                            <tr>
                                <td class="perftd">Time spent so far:</td>
                                <td class="perftd" id="st">00:00:00</td>
                            </tr>
                        </tbody>
                    </table>
                </center>
            </div>
            <br/>
            <div style="width: 100%; background-color: #FFF; text-align:center; display: block; ">
                <center>
                    <table style="margin: 10px; width: 200px;">
                        <thead style="width: 200px;">
                            <td colspan="2"><b>Performance status:</b></td>
                        </thead>
                        <tbody>
                            <tr>
                                <td class="perftd">Sampes per sec:</td>
                                <td class="perftd" id="ps">0.0/sec</td>
                            </tr>
                            <tr>
                                <td class="perftd">Batches per sec:</td>
                                <td class="perftd" id="pb">0.0/sec</td>
                            </tr>
                            <tr>
                                <td class="perftd">Iteration time:</td>
                                <td class="perftd" id="pt">0 ms</td>
                            </tr>
                        </tbody>
                    </table>
                </center>
            </div>
        </div>

            <!-- Right view panel -->
        <div style="right: 10px; width: 400px; position: absolute;" class="viewpanel" id="viewport">
            <div style='position: relative; top: 45%; height: 40px; margin: 0 auto;' id='hint'><b>&lt; Click on any node for detailed report</b></div>
        </div>
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
object Flow extends Flow_Scope0.Flow
              /*
                  -- GENERATED --
                  DATE: Fri May 18 19:33:53 PDT 2018
                  SOURCE: C:/develop/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/flow/Flow.scala.html
                  HASH: 16da75957d6324a8afc14f57af86666402e62d06
                  MATRIX: 634->0|1990->1328|2019->1329|2061->1343|2219->1474|2248->1475|2285->1485|2317->1489|2346->1490|2388->1504|2510->1599|2539->1600|2576->1610|2611->1617|2640->1618|2682->1632|2869->1792|2898->1793|2935->1803|2973->1813|3002->1814|3044->1828|3166->1923|3195->1924|3232->1934|3276->1950|3305->1951|3347->1965|3434->2025|3463->2026|3500->2036|3537->2045|3566->2046|3608->2060|4105->2530|4134->2531|4171->2541|4210->2552|4239->2553|4281->2567|4424->2683|4453->2684|4492->2696|4528->2704|4557->2705|4599->2719|4753->2846|4782->2847|4821->2859|4859->2869|4888->2870|4930->2884|5026->2952|5056->2953|5096->2965|5135->2975|5165->2976|5208->2990|5260->3014|5290->3015|5330->3027|5365->3033|5395->3034|5438->3048|5492->3074|5522->3075|5562->3087|5614->3110|5644->3111|5687->3125|5837->3247|5867->3248|5907->3260|5947->3271|5977->3272|6020->3286|6145->3383|6175->3384|6215->3396|6250->3402|6280->3403|6323->3417|6376->3442|6406->3443|6446->3455|6480->3460|6510->3461|6553->3475|6666->3560|6696->3561|6736->3573|6773->3581|6803->3582|6846->3596|6934->3656|6964->3657|7004->3669|7044->3680|7074->3681|7117->3695|7204->3754|7234->3755|7274->3767|7318->3782|7348->3783|7391->3797|7549->3927|7579->3928|7617->3938
                  LINES: 25->1|64->40|64->40|65->41|69->45|69->45|70->46|70->46|70->46|71->47|74->50|74->50|75->51|75->51|75->51|76->52|81->57|81->57|82->58|82->58|82->58|83->59|86->62|86->62|87->63|87->63|87->63|88->64|90->66|90->66|91->67|91->67|91->67|92->68|107->83|107->83|108->84|108->84|108->84|109->85|113->89|113->89|115->91|115->91|115->91|116->92|120->96|120->96|122->98|122->98|122->98|123->99|125->101|125->101|127->103|127->103|127->103|128->104|129->105|129->105|131->107|131->107|131->107|132->108|133->109|133->109|135->111|135->111|135->111|136->112|140->116|140->116|142->118|142->118|142->118|143->119|146->122|146->122|148->124|148->124|148->124|149->125|150->126|150->126|152->128|152->128|152->128|153->129|156->132|156->132|158->134|158->134|158->134|159->135|161->137|161->137|163->139|163->139|163->139|164->140|166->142|166->142|168->144|168->144|168->144|169->145|173->149|173->149|174->150
                  -- GENERATED --
              */
          