
package org.deeplearning4j.ui.views.html.tsne

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object Tsne_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class Tsne extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template0[play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply():play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.1*/("""<!DOCTYPE html>

<!--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~ Copyright (c) 2015-2018 Skymind, Inc.
  ~
  ~ This program and the accompanying materials are made available under the
  ~ terms of the Apache License, Version 2.0 which is available at
  ~ https://www.apache.org/licenses/LICENSE-2.0.
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
  ~ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
  ~ License for the specific language governing permissions and limitations
  ~ under the License.
  ~
  ~ SPDX-License-Identifier: Apache-2.0
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-->

<html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>T-SNE renders</title>
            <!-- jQuery -->
        <script src="/assets/webjars/jquery/2.2.0/jquery.min.js"></script>

        <link href='/assets/legacy/roboto.css' rel='stylesheet' type='text/css'>
        <link src="/assets/webjars/bootstrap/2.3.1/css/bootstrap.min.css" rel="stylesheet" ></link>

            <!-- Latest compiled and minified JavaScript -->
        """),format.raw/*30.69*/("""
        """),format.raw/*31.9*/("""<script src="/assets/webjars/bootstrap/2.3.1/js/bootstrap.min.js"></script>


            <!-- d3 -->
        <script src="/assets/webjars/d3js/3.3.5/d3.min.js" charset="utf-8"></script>


        <script src="/assets/legacy/jquery-fileupload.js"></script>

            <!-- Booststrap Notify plugin-->
        <script src="/assets/webjars/bootstrap-notify/3.1.3-1/bootstrap-notify.min.js"></script>

        <script src="/assets/legacy/common.js"></script>

            <!-- dl4j plot setup -->
        <script src="/assets/legacy/renderTsne.js"></script>


        <style>
        .hd """),format.raw/*50.13*/("""{"""),format.raw/*50.14*/("""
            """),format.raw/*51.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*54.9*/("""}"""),format.raw/*54.10*/("""

        """),format.raw/*56.9*/(""".block """),format.raw/*56.16*/("""{"""),format.raw/*56.17*/("""
            """),format.raw/*57.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*62.9*/("""}"""),format.raw/*62.10*/("""

        """),format.raw/*64.9*/(""".hd-small """),format.raw/*64.19*/("""{"""),format.raw/*64.20*/("""
            """),format.raw/*65.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*68.9*/("""}"""),format.raw/*68.10*/("""

        """),format.raw/*70.9*/("""body """),format.raw/*70.14*/("""{"""),format.raw/*70.15*/("""
            """),format.raw/*71.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*75.9*/("""}"""),format.raw/*75.10*/("""

        """),format.raw/*77.9*/("""#wrap """),format.raw/*77.15*/("""{"""),format.raw/*77.16*/("""
            """),format.raw/*78.13*/("""width: 800px;
            margin-left: auto;
            margin-right: auto;
        """),format.raw/*81.9*/("""}"""),format.raw/*81.10*/("""

        """),format.raw/*83.9*/("""#embed """),format.raw/*83.16*/("""{"""),format.raw/*83.17*/("""
            """),format.raw/*84.13*/("""margin-top: 10px;
        """),format.raw/*85.9*/("""}"""),format.raw/*85.10*/("""

        """),format.raw/*87.9*/("""h1 """),format.raw/*87.12*/("""{"""),format.raw/*87.13*/("""
            """),format.raw/*88.13*/("""text-align: center;
            font-weight: normal;
        """),format.raw/*90.9*/("""}"""),format.raw/*90.10*/("""

        """),format.raw/*92.9*/(""".tt """),format.raw/*92.13*/("""{"""),format.raw/*92.14*/("""
            """),format.raw/*93.13*/("""margin-top: 10px;
            background-color: #EEE;
            border-bottom: 1px solid #333;
            padding: 5px;
        """),format.raw/*97.9*/("""}"""),format.raw/*97.10*/("""

        """),format.raw/*99.9*/(""".txth """),format.raw/*99.15*/("""{"""),format.raw/*99.16*/("""
            """),format.raw/*100.13*/("""color: #F55;
        """),format.raw/*101.9*/("""}"""),format.raw/*101.10*/("""

        """),format.raw/*103.9*/(""".cit """),format.raw/*103.14*/("""{"""),format.raw/*103.15*/("""
            """),format.raw/*104.13*/("""font-family: courier;
            padding-left: 20px;
            font-size: 14px;
        """),format.raw/*107.9*/("""}"""),format.raw/*107.10*/("""
        """),format.raw/*108.9*/("""</style>

        <script>
        $(document).ready(function () """),format.raw/*111.39*/("""{"""),format.raw/*111.40*/("""
            """),format.raw/*112.13*/("""$('#filenamebutton').click(function () """),format.raw/*112.52*/("""{"""),format.raw/*112.53*/("""
                """),format.raw/*113.17*/("""document.getElementById('form').reset();
                $('#form').hide();
                var filename = $('#filename').val();
                $('#filename').val('');
                updateFileName(filename);
                drawTsne();
            """),format.raw/*119.13*/("""}"""),format.raw/*119.14*/(""");

            $('#form').fileUpload("""),format.raw/*121.35*/("""{"""),format.raw/*121.36*/("""
                """),format.raw/*122.17*/("""success: function (data, textStatus, jqXHR) """),format.raw/*122.61*/("""{"""),format.raw/*122.62*/("""
                    """),format.raw/*123.21*/("""var fullPath = document.getElementById('form').value;
                    var filename = data['name'];
                    if (fullPath) """),format.raw/*125.35*/("""{"""),format.raw/*125.36*/("""
                        """),format.raw/*126.25*/("""var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
                        var filename = fullPath.substring(startIndex);
                        if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) """),format.raw/*128.90*/("""{"""),format.raw/*128.91*/("""
                            """),format.raw/*129.29*/("""filename = filename.substring(1);
                        """),format.raw/*130.25*/("""}"""),format.raw/*130.26*/("""
                    """),format.raw/*131.21*/("""}"""),format.raw/*131.22*/("""

                    """),format.raw/*133.21*/("""document.getElementById('form').reset();

                    updateFileName(filename);
                    drawTsne();
                """),format.raw/*137.17*/("""}"""),format.raw/*137.18*/(""", error: function (err) """),format.raw/*137.42*/("""{"""),format.raw/*137.43*/("""
                    """),format.raw/*138.21*/("""console.log(err);
                    drawTsne();
                """),format.raw/*140.17*/("""}"""),format.raw/*140.18*/("""
            """),format.raw/*141.13*/("""}"""),format.raw/*141.14*/(""");


            function updateFileName(name) """),format.raw/*144.43*/("""{"""),format.raw/*144.44*/("""
                """),format.raw/*145.17*/("""$.ajax("""),format.raw/*145.24*/("""{"""),format.raw/*145.25*/("""
                    """),format.raw/*146.21*/("""url: '/tsne/upload',
                    type: 'POST',
                    dataType: 'json',
                    data: JSON.stringify("""),format.raw/*149.42*/("""{"""),format.raw/*149.43*/(""""url": name"""),format.raw/*149.54*/("""}"""),format.raw/*149.55*/("""),
                    cache: false,
                    success: function (data, textStatus, jqXHR) """),format.raw/*151.65*/("""{"""),format.raw/*151.66*/("""
                        """),format.raw/*152.25*/("""setSessionId("UploadedFile");
                        drawTsne();
                    """),format.raw/*154.21*/("""}"""),format.raw/*154.22*/(""",
                    error: function (jqXHR, textStatus, errorThrown) """),format.raw/*155.70*/("""{"""),format.raw/*155.71*/("""
                        """),format.raw/*156.25*/("""// Handle errors here
                        console.log('ERRORS: ' + textStatus);
                        drawTsne();
                    """),format.raw/*159.21*/("""}"""),format.raw/*159.22*/("""
                """),format.raw/*160.17*/("""}"""),format.raw/*160.18*/(""");
            """),format.raw/*161.13*/("""}"""),format.raw/*161.14*/("""


        """),format.raw/*164.9*/("""}"""),format.raw/*164.10*/(""") ;

    </script>

    </head>

    <body>
        <table style="width: 100%;
            padding: 5px;" class="hd">
            <tbody>
                <tr>
                    <td style="width: 48px;"><a href="/"><img src="/assets/legacy/deeplearning4j.img" border="0"/></a></td>
                    <td>DeepLearning4j UI</td>
                    <td style="width: 512px;
                        text-align: right;" class="hd-small">&nbsp; Available sessions:
                        <select class="selectpicker" id="sessionSelect" onchange="selectNewSession()" style="color: #000000;
                            display: inline-block;
                            width: 256px;">
                            <option value="0" selected="selected">Pick a session to track</option>
                        </select>
                    </td>
                    <td style="width: 256px;">&nbsp; <!-- placeholder for future use --></td>
                </tr>
            </tbody>
        </table>

        <br />
        <div style="text-align: center">
            <div id="embed" style="display: inline-block;
                width: 1024px;
                height: 700px;
                border: 1px solid #DEDEDE;"></div>
        </div>
        <br/>
        <br/>
        <div style="text-align: center;
            width: 100%;
            position: fixed;
            bottom: 0px;
            left: 0px;
            margin-bottom: 15px;">
            <div style="display: inline-block;
                margin-right: 48px;">
                <h5>Upload a file to UI server.</h5>
                <form encType="multipart/form-data" action="/tsne/upload" method="POST" id="form">
                    <div>

                        <input name="file" type="file" style="width: 300px;
                            display: inline-block;" />
                        <input type="submit" value="Upload file" style="display: inline-block;"/>

                    </div>
                </form>
            </div>

            """),format.raw/*219.53*/("""
                """),format.raw/*220.96*/("""
                """),format.raw/*221.42*/("""
                    """),format.raw/*222.59*/("""
                    """),format.raw/*223.68*/("""
                """),format.raw/*224.27*/("""
            """),format.raw/*225.23*/("""
        """),format.raw/*226.9*/("""</div>
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
object Tsne extends Tsne_Scope0.Tsne
              /*
                  -- GENERATED --
                  DATE: Tue Jan 22 16:29:57 AEDT 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/tsne/Tsne.scala.html
                  HASH: 80552117a6bd6eb97112ce7dfa4de23ba28d40c9
                  MATRIX: 634->0|1937->1335|1974->1345|2608->1951|2637->1952|2679->1966|2801->2061|2830->2062|2869->2074|2904->2081|2933->2082|2975->2096|3162->2256|3191->2257|3230->2269|3268->2279|3297->2280|3339->2294|3461->2389|3490->2390|3529->2402|3562->2407|3591->2408|3633->2422|3791->2553|3820->2554|3859->2566|3893->2572|3922->2573|3964->2587|4079->2675|4108->2676|4147->2688|4182->2695|4211->2696|4253->2710|4307->2737|4336->2738|4375->2750|4406->2753|4435->2754|4477->2768|4567->2831|4596->2832|4635->2844|4667->2848|4696->2849|4738->2863|4900->2998|4929->2999|4968->3011|5002->3017|5031->3018|5074->3032|5124->3054|5154->3055|5194->3067|5228->3072|5258->3073|5301->3087|5423->3181|5453->3182|5491->3192|5588->3260|5618->3261|5661->3275|5729->3314|5759->3315|5806->3333|6092->3590|6122->3591|6191->3631|6221->3632|6268->3650|6341->3694|6371->3695|6422->3717|6590->3856|6620->3857|6675->3883|6971->4150|7001->4151|7060->4181|7148->4240|7178->4241|7229->4263|7259->4264|7312->4288|7481->4428|7511->4429|7564->4453|7594->4454|7645->4476|7742->4544|7772->4545|7815->4559|7845->4560|7924->4610|7954->4611|8001->4629|8037->4636|8067->4637|8118->4659|8284->4796|8314->4797|8354->4808|8384->4809|8516->4912|8546->4913|8601->4939|8718->5027|8748->5028|8849->5100|8879->5101|8934->5127|9106->5270|9136->5271|9183->5289|9213->5290|9258->5306|9288->5307|9330->5321|9360->5322|11465->7438|11512->7535|11559->7578|11610->7638|11661->7707|11708->7735|11751->7759|11789->7769
                  LINES: 25->1|54->30|55->31|74->50|74->50|75->51|78->54|78->54|80->56|80->56|80->56|81->57|86->62|86->62|88->64|88->64|88->64|89->65|92->68|92->68|94->70|94->70|94->70|95->71|99->75|99->75|101->77|101->77|101->77|102->78|105->81|105->81|107->83|107->83|107->83|108->84|109->85|109->85|111->87|111->87|111->87|112->88|114->90|114->90|116->92|116->92|116->92|117->93|121->97|121->97|123->99|123->99|123->99|124->100|125->101|125->101|127->103|127->103|127->103|128->104|131->107|131->107|132->108|135->111|135->111|136->112|136->112|136->112|137->113|143->119|143->119|145->121|145->121|146->122|146->122|146->122|147->123|149->125|149->125|150->126|152->128|152->128|153->129|154->130|154->130|155->131|155->131|157->133|161->137|161->137|161->137|161->137|162->138|164->140|164->140|165->141|165->141|168->144|168->144|169->145|169->145|169->145|170->146|173->149|173->149|173->149|173->149|175->151|175->151|176->152|178->154|178->154|179->155|179->155|180->156|183->159|183->159|184->160|184->160|185->161|185->161|188->164|188->164|243->219|244->220|245->221|246->222|247->223|248->224|249->225|250->226
                  -- GENERATED --
              */
          