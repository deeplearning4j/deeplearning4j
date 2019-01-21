
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
        """),format.raw/*24.71*/("""
        """),format.raw/*25.9*/("""<script src="/assets/webjars/jquery/1.9.1/jquery.min.js"></script>

        <link href='/assets/legacy/roboto.css' rel='stylesheet' type='text/css'>

            <!-- Latest compiled and minified CSS -->
        """),format.raw/*30.78*/("""
        """),format.raw/*31.9*/("""<script src="/assets/webjars/bootstrap/2.3.1/css/bootstrap.min.css"></script>

            <!-- Optional theme -->
        <link rel="stylesheet" href="/assets/legacy/bootstrap-theme.min.css" />

            <!-- Latest compiled and minified JavaScript -->
        """),format.raw/*37.69*/("""
        """),format.raw/*38.9*/("""<script src="/assets/webjars/bootstrap/2.3.1/js/bootstrap.min.js"></script>


            <!-- d3 -->
        <script src="/assets/webjars/d3js/3.3.5/d3.min.js" charset="utf-8"></script>


        <script src="/assets/legacy/jquery-fileupload.js"></script>

            <!-- Booststrap Notify plugin-->
        <script src="/assets/webjars/bootstrap-notify/3.1.3-1/bootstrap-notify.min.js"></script>

        <script src="/assets/legacy/common.js"></script>

            <!-- dl4j plot setup -->
        <script src="/assets/legacy/renderTsne.js"></script>


        <style>
        .hd """),format.raw/*57.13*/("""{"""),format.raw/*57.14*/("""
            """),format.raw/*58.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*61.9*/("""}"""),format.raw/*61.10*/("""

        """),format.raw/*63.9*/(""".block """),format.raw/*63.16*/("""{"""),format.raw/*63.17*/("""
            """),format.raw/*64.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*69.9*/("""}"""),format.raw/*69.10*/("""

        """),format.raw/*71.9*/(""".hd-small """),format.raw/*71.19*/("""{"""),format.raw/*71.20*/("""
            """),format.raw/*72.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*75.9*/("""}"""),format.raw/*75.10*/("""

        """),format.raw/*77.9*/("""body """),format.raw/*77.14*/("""{"""),format.raw/*77.15*/("""
            """),format.raw/*78.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*82.9*/("""}"""),format.raw/*82.10*/("""

        """),format.raw/*84.9*/("""#wrap """),format.raw/*84.15*/("""{"""),format.raw/*84.16*/("""
            """),format.raw/*85.13*/("""width: 800px;
            margin-left: auto;
            margin-right: auto;
        """),format.raw/*88.9*/("""}"""),format.raw/*88.10*/("""

        """),format.raw/*90.9*/("""#embed """),format.raw/*90.16*/("""{"""),format.raw/*90.17*/("""
            """),format.raw/*91.13*/("""margin-top: 10px;
        """),format.raw/*92.9*/("""}"""),format.raw/*92.10*/("""

        """),format.raw/*94.9*/("""h1 """),format.raw/*94.12*/("""{"""),format.raw/*94.13*/("""
            """),format.raw/*95.13*/("""text-align: center;
            font-weight: normal;
        """),format.raw/*97.9*/("""}"""),format.raw/*97.10*/("""

        """),format.raw/*99.9*/(""".tt """),format.raw/*99.13*/("""{"""),format.raw/*99.14*/("""
            """),format.raw/*100.13*/("""margin-top: 10px;
            background-color: #EEE;
            border-bottom: 1px solid #333;
            padding: 5px;
        """),format.raw/*104.9*/("""}"""),format.raw/*104.10*/("""

        """),format.raw/*106.9*/(""".txth """),format.raw/*106.15*/("""{"""),format.raw/*106.16*/("""
            """),format.raw/*107.13*/("""color: #F55;
        """),format.raw/*108.9*/("""}"""),format.raw/*108.10*/("""

        """),format.raw/*110.9*/(""".cit """),format.raw/*110.14*/("""{"""),format.raw/*110.15*/("""
            """),format.raw/*111.13*/("""font-family: courier;
            padding-left: 20px;
            font-size: 14px;
        """),format.raw/*114.9*/("""}"""),format.raw/*114.10*/("""
        """),format.raw/*115.9*/("""</style>

        <script>
        $(document).ready(function () """),format.raw/*118.39*/("""{"""),format.raw/*118.40*/("""
            """),format.raw/*119.13*/("""$('#filenamebutton').click(function () """),format.raw/*119.52*/("""{"""),format.raw/*119.53*/("""
                """),format.raw/*120.17*/("""document.getElementById('form').reset();
                $('#form').hide();
                var filename = $('#filename').val();
                $('#filename').val('');
                updateFileName(filename);
                drawTsne();
            """),format.raw/*126.13*/("""}"""),format.raw/*126.14*/(""");

            $('#form').fileUpload("""),format.raw/*128.35*/("""{"""),format.raw/*128.36*/("""
                """),format.raw/*129.17*/("""success: function (data, textStatus, jqXHR) """),format.raw/*129.61*/("""{"""),format.raw/*129.62*/("""
                    """),format.raw/*130.21*/("""var fullPath = document.getElementById('form').value;
                    var filename = data['name'];
                    if (fullPath) """),format.raw/*132.35*/("""{"""),format.raw/*132.36*/("""
                        """),format.raw/*133.25*/("""var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
                        var filename = fullPath.substring(startIndex);
                        if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) """),format.raw/*135.90*/("""{"""),format.raw/*135.91*/("""
                            """),format.raw/*136.29*/("""filename = filename.substring(1);
                        """),format.raw/*137.25*/("""}"""),format.raw/*137.26*/("""
                    """),format.raw/*138.21*/("""}"""),format.raw/*138.22*/("""

                    """),format.raw/*140.21*/("""document.getElementById('form').reset();

                    updateFileName(filename);
                    drawTsne();
                """),format.raw/*144.17*/("""}"""),format.raw/*144.18*/(""", error: function (err) """),format.raw/*144.42*/("""{"""),format.raw/*144.43*/("""
                    """),format.raw/*145.21*/("""console.log(err);
                    drawTsne();
                """),format.raw/*147.17*/("""}"""),format.raw/*147.18*/("""
            """),format.raw/*148.13*/("""}"""),format.raw/*148.14*/(""");


            function updateFileName(name) """),format.raw/*151.43*/("""{"""),format.raw/*151.44*/("""
                """),format.raw/*152.17*/("""$.ajax("""),format.raw/*152.24*/("""{"""),format.raw/*152.25*/("""
                    """),format.raw/*153.21*/("""url: '/tsne/upload',
                    type: 'POST',
                    dataType: 'json',
                    data: JSON.stringify("""),format.raw/*156.42*/("""{"""),format.raw/*156.43*/(""""url": name"""),format.raw/*156.54*/("""}"""),format.raw/*156.55*/("""),
                    cache: false,
                    success: function (data, textStatus, jqXHR) """),format.raw/*158.65*/("""{"""),format.raw/*158.66*/("""
                        """),format.raw/*159.25*/("""setSessionId("UploadedFile");
                        drawTsne();
                    """),format.raw/*161.21*/("""}"""),format.raw/*161.22*/(""",
                    error: function (jqXHR, textStatus, errorThrown) """),format.raw/*162.70*/("""{"""),format.raw/*162.71*/("""
                        """),format.raw/*163.25*/("""// Handle errors here
                        console.log('ERRORS: ' + textStatus);
                        drawTsne();
                    """),format.raw/*166.21*/("""}"""),format.raw/*166.22*/("""
                """),format.raw/*167.17*/("""}"""),format.raw/*167.18*/(""");
            """),format.raw/*168.13*/("""}"""),format.raw/*168.14*/("""


        """),format.raw/*171.9*/("""}"""),format.raw/*171.10*/(""") ;

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

            """),format.raw/*226.53*/("""
                """),format.raw/*227.96*/("""
                """),format.raw/*228.42*/("""
                    """),format.raw/*229.59*/("""
                    """),format.raw/*230.68*/("""
                """),format.raw/*231.27*/("""
            """),format.raw/*232.23*/("""
        """),format.raw/*233.9*/("""</div>
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
                  DATE: Mon Jan 21 12:53:56 AEDT 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/tsne/Tsne.scala.html
                  HASH: e0d20592e4a13f09d398f8704a35eb69a2af3785
                  MATRIX: 634->0|1612->1012|1649->1022|1894->1308|1931->1318|2230->1649|2267->1659|2901->2265|2930->2266|2972->2280|3094->2375|3123->2376|3162->2388|3197->2395|3226->2396|3268->2410|3455->2570|3484->2571|3523->2583|3561->2593|3590->2594|3632->2608|3754->2703|3783->2704|3822->2716|3855->2721|3884->2722|3926->2736|4084->2867|4113->2868|4152->2880|4186->2886|4215->2887|4257->2901|4372->2989|4401->2990|4440->3002|4475->3009|4504->3010|4546->3024|4600->3051|4629->3052|4668->3064|4699->3067|4728->3068|4770->3082|4860->3145|4889->3146|4928->3158|4960->3162|4989->3163|5032->3177|5195->3312|5225->3313|5265->3325|5300->3331|5330->3332|5373->3346|5423->3368|5453->3369|5493->3381|5527->3386|5557->3387|5600->3401|5722->3495|5752->3496|5790->3506|5887->3574|5917->3575|5960->3589|6028->3628|6058->3629|6105->3647|6391->3904|6421->3905|6490->3945|6520->3946|6567->3964|6640->4008|6670->4009|6721->4031|6889->4170|6919->4171|6974->4197|7270->4464|7300->4465|7359->4495|7447->4554|7477->4555|7528->4577|7558->4578|7611->4602|7780->4742|7810->4743|7863->4767|7893->4768|7944->4790|8041->4858|8071->4859|8114->4873|8144->4874|8223->4924|8253->4925|8300->4943|8336->4950|8366->4951|8417->4973|8583->5110|8613->5111|8653->5122|8683->5123|8815->5226|8845->5227|8900->5253|9017->5341|9047->5342|9148->5414|9178->5415|9233->5441|9405->5584|9435->5585|9482->5603|9512->5604|9557->5620|9587->5621|9629->5635|9659->5636|11764->7752|11811->7849|11858->7892|11909->7952|11960->8021|12007->8049|12050->8073|12088->8083
                  LINES: 25->1|48->24|49->25|54->30|55->31|61->37|62->38|81->57|81->57|82->58|85->61|85->61|87->63|87->63|87->63|88->64|93->69|93->69|95->71|95->71|95->71|96->72|99->75|99->75|101->77|101->77|101->77|102->78|106->82|106->82|108->84|108->84|108->84|109->85|112->88|112->88|114->90|114->90|114->90|115->91|116->92|116->92|118->94|118->94|118->94|119->95|121->97|121->97|123->99|123->99|123->99|124->100|128->104|128->104|130->106|130->106|130->106|131->107|132->108|132->108|134->110|134->110|134->110|135->111|138->114|138->114|139->115|142->118|142->118|143->119|143->119|143->119|144->120|150->126|150->126|152->128|152->128|153->129|153->129|153->129|154->130|156->132|156->132|157->133|159->135|159->135|160->136|161->137|161->137|162->138|162->138|164->140|168->144|168->144|168->144|168->144|169->145|171->147|171->147|172->148|172->148|175->151|175->151|176->152|176->152|176->152|177->153|180->156|180->156|180->156|180->156|182->158|182->158|183->159|185->161|185->161|186->162|186->162|187->163|190->166|190->166|191->167|191->167|192->168|192->168|195->171|195->171|250->226|251->227|252->228|253->229|254->230|255->231|256->232|257->233
                  -- GENERATED --
              */
          