
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
                  DATE: Tue Oct 25 15:17:44 PDT 2016
                  SOURCE: /Users/ejunprung/skymind-ui/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/flow/Flow.scala.html
                  HASH: 41d1e77c9fda2183d9dab7ab8d24d8c46d3256a6
                  MATRIX: 634->0|1951->1289|1980->1290|2021->1303|2175->1430|2204->1431|2240->1440|2272->1444|2301->1445|2342->1458|2461->1550|2490->1551|2526->1560|2561->1567|2590->1568|2631->1581|2813->1736|2842->1737|2878->1746|2916->1756|2945->1757|2986->1770|3105->1862|3134->1863|3170->1872|3214->1888|3243->1889|3284->1902|3369->1960|3398->1961|3434->1970|3471->1979|3500->1980|3541->1993|4023->2448|4052->2449|4088->2458|4127->2469|4156->2470|4197->2483|4336->2595|4365->2596|4402->2606|4438->2614|4467->2615|4508->2628|4658->2751|4687->2752|4724->2762|4762->2772|4791->2773|4832->2786|4926->2852|4956->2853|4994->2863|5033->2873|5063->2874|5105->2887|5156->2910|5186->2911|5224->2921|5259->2927|5289->2928|5331->2941|5384->2966|5414->2967|5452->2977|5504->3000|5534->3001|5576->3014|5722->3132|5752->3133|5790->3143|5830->3154|5860->3155|5902->3168|6024->3262|6054->3263|6092->3273|6127->3279|6157->3280|6199->3293|6251->3317|6281->3318|6319->3328|6353->3333|6383->3334|6425->3347|6535->3429|6565->3430|6603->3440|6640->3448|6670->3449|6712->3462|6798->3520|6828->3521|6866->3531|6906->3542|6936->3543|6978->3556|7063->3613|7093->3614|7131->3624|7175->3639|7205->3640|7247->3653|7401->3779|7431->3780|7468->3789
                  LINES: 25->1|64->40|64->40|65->41|69->45|69->45|70->46|70->46|70->46|71->47|74->50|74->50|75->51|75->51|75->51|76->52|81->57|81->57|82->58|82->58|82->58|83->59|86->62|86->62|87->63|87->63|87->63|88->64|90->66|90->66|91->67|91->67|91->67|92->68|107->83|107->83|108->84|108->84|108->84|109->85|113->89|113->89|115->91|115->91|115->91|116->92|120->96|120->96|122->98|122->98|122->98|123->99|125->101|125->101|127->103|127->103|127->103|128->104|129->105|129->105|131->107|131->107|131->107|132->108|133->109|133->109|135->111|135->111|135->111|136->112|140->116|140->116|142->118|142->118|142->118|143->119|146->122|146->122|148->124|148->124|148->124|149->125|150->126|150->126|152->128|152->128|152->128|153->129|156->132|156->132|158->134|158->134|158->134|159->135|161->137|161->137|163->139|163->139|163->139|164->140|166->142|166->142|168->144|168->144|168->144|169->145|173->149|173->149|174->150
                  -- GENERATED --
              */
          