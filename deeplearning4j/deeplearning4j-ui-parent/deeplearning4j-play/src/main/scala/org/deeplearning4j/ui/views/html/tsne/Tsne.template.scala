
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
        <script src="/assets/webjars/bootstrap/2.3.1/css/bootstrap.min.css"></script>

            <!-- Latest compiled and minified JavaScript -->
        """),format.raw/*31.69*/("""
        """),format.raw/*32.9*/("""<script src="/assets/webjars/bootstrap/2.3.1/js/bootstrap.min.js"></script>


            <!-- d3 -->
        <script src="/assets/webjars/d3js/3.3.5/d3.min.js" charset="utf-8"></script>


        <script src="/assets/legacy/jquery-fileupload.js"></script>

            <!-- Booststrap Notify plugin-->
        <script src="/assets/webjars/bootstrap-notify/3.1.3-1/bootstrap-notify.min.js"></script>

        <script src="/assets/legacy/common.js"></script>

            <!-- dl4j plot setup -->
        <script src="/assets/legacy/renderTsne.js"></script>


        <style>
        .hd """),format.raw/*51.13*/("""{"""),format.raw/*51.14*/("""
            """),format.raw/*52.13*/("""background-color: #000000;
            font-size: 18px;
            color: #FFFFFF;
        """),format.raw/*55.9*/("""}"""),format.raw/*55.10*/("""

        """),format.raw/*57.9*/(""".block """),format.raw/*57.16*/("""{"""),format.raw/*57.17*/("""
            """),format.raw/*58.13*/("""width: 250px;
            height: 350px;
            display: inline-block;
            border: 1px solid #DEDEDE;
            margin-right: 64px;
        """),format.raw/*63.9*/("""}"""),format.raw/*63.10*/("""

        """),format.raw/*65.9*/(""".hd-small """),format.raw/*65.19*/("""{"""),format.raw/*65.20*/("""
            """),format.raw/*66.13*/("""background-color: #000000;
            font-size: 14px;
            color: #FFFFFF;
        """),format.raw/*69.9*/("""}"""),format.raw/*69.10*/("""

        """),format.raw/*71.9*/("""body """),format.raw/*71.14*/("""{"""),format.raw/*71.15*/("""
            """),format.raw/*72.13*/("""font-family: 'Roboto', sans-serif;
            color: #333;
            font-weight: 300;
            font-size: 16px;
        """),format.raw/*76.9*/("""}"""),format.raw/*76.10*/("""

        """),format.raw/*78.9*/("""#wrap """),format.raw/*78.15*/("""{"""),format.raw/*78.16*/("""
            """),format.raw/*79.13*/("""width: 800px;
            margin-left: auto;
            margin-right: auto;
        """),format.raw/*82.9*/("""}"""),format.raw/*82.10*/("""

        """),format.raw/*84.9*/("""#embed """),format.raw/*84.16*/("""{"""),format.raw/*84.17*/("""
            """),format.raw/*85.13*/("""margin-top: 10px;
        """),format.raw/*86.9*/("""}"""),format.raw/*86.10*/("""

        """),format.raw/*88.9*/("""h1 """),format.raw/*88.12*/("""{"""),format.raw/*88.13*/("""
            """),format.raw/*89.13*/("""text-align: center;
            font-weight: normal;
        """),format.raw/*91.9*/("""}"""),format.raw/*91.10*/("""

        """),format.raw/*93.9*/(""".tt """),format.raw/*93.13*/("""{"""),format.raw/*93.14*/("""
            """),format.raw/*94.13*/("""margin-top: 10px;
            background-color: #EEE;
            border-bottom: 1px solid #333;
            padding: 5px;
        """),format.raw/*98.9*/("""}"""),format.raw/*98.10*/("""

        """),format.raw/*100.9*/(""".txth """),format.raw/*100.15*/("""{"""),format.raw/*100.16*/("""
            """),format.raw/*101.13*/("""color: #F55;
        """),format.raw/*102.9*/("""}"""),format.raw/*102.10*/("""

        """),format.raw/*104.9*/(""".cit """),format.raw/*104.14*/("""{"""),format.raw/*104.15*/("""
            """),format.raw/*105.13*/("""font-family: courier;
            padding-left: 20px;
            font-size: 14px;
        """),format.raw/*108.9*/("""}"""),format.raw/*108.10*/("""
        """),format.raw/*109.9*/("""</style>

        <script>
        $(document).ready(function () """),format.raw/*112.39*/("""{"""),format.raw/*112.40*/("""
            """),format.raw/*113.13*/("""$('#filenamebutton').click(function () """),format.raw/*113.52*/("""{"""),format.raw/*113.53*/("""
                """),format.raw/*114.17*/("""document.getElementById('form').reset();
                $('#form').hide();
                var filename = $('#filename').val();
                $('#filename').val('');
                updateFileName(filename);
                drawTsne();
            """),format.raw/*120.13*/("""}"""),format.raw/*120.14*/(""");

            $('#form').fileUpload("""),format.raw/*122.35*/("""{"""),format.raw/*122.36*/("""
                """),format.raw/*123.17*/("""success: function (data, textStatus, jqXHR) """),format.raw/*123.61*/("""{"""),format.raw/*123.62*/("""
                    """),format.raw/*124.21*/("""var fullPath = document.getElementById('form').value;
                    var filename = data['name'];
                    if (fullPath) """),format.raw/*126.35*/("""{"""),format.raw/*126.36*/("""
                        """),format.raw/*127.25*/("""var startIndex = (fullPath.indexOf('\\') >= 0 ? fullPath.lastIndexOf('\\') : fullPath.lastIndexOf('/'));
                        var filename = fullPath.substring(startIndex);
                        if (filename.indexOf('\\') === 0 || filename.indexOf('/') === 0) """),format.raw/*129.90*/("""{"""),format.raw/*129.91*/("""
                            """),format.raw/*130.29*/("""filename = filename.substring(1);
                        """),format.raw/*131.25*/("""}"""),format.raw/*131.26*/("""
                    """),format.raw/*132.21*/("""}"""),format.raw/*132.22*/("""

                    """),format.raw/*134.21*/("""document.getElementById('form').reset();

                    updateFileName(filename);
                    drawTsne();
                """),format.raw/*138.17*/("""}"""),format.raw/*138.18*/(""", error: function (err) """),format.raw/*138.42*/("""{"""),format.raw/*138.43*/("""
                    """),format.raw/*139.21*/("""console.log(err);
                    drawTsne();
                """),format.raw/*141.17*/("""}"""),format.raw/*141.18*/("""
            """),format.raw/*142.13*/("""}"""),format.raw/*142.14*/(""");


            function updateFileName(name) """),format.raw/*145.43*/("""{"""),format.raw/*145.44*/("""
                """),format.raw/*146.17*/("""$.ajax("""),format.raw/*146.24*/("""{"""),format.raw/*146.25*/("""
                    """),format.raw/*147.21*/("""url: '/tsne/upload',
                    type: 'POST',
                    dataType: 'json',
                    data: JSON.stringify("""),format.raw/*150.42*/("""{"""),format.raw/*150.43*/(""""url": name"""),format.raw/*150.54*/("""}"""),format.raw/*150.55*/("""),
                    cache: false,
                    success: function (data, textStatus, jqXHR) """),format.raw/*152.65*/("""{"""),format.raw/*152.66*/("""
                        """),format.raw/*153.25*/("""setSessionId("UploadedFile");
                        drawTsne();
                    """),format.raw/*155.21*/("""}"""),format.raw/*155.22*/(""",
                    error: function (jqXHR, textStatus, errorThrown) """),format.raw/*156.70*/("""{"""),format.raw/*156.71*/("""
                        """),format.raw/*157.25*/("""// Handle errors here
                        console.log('ERRORS: ' + textStatus);
                        drawTsne();
                    """),format.raw/*160.21*/("""}"""),format.raw/*160.22*/("""
                """),format.raw/*161.17*/("""}"""),format.raw/*161.18*/(""");
            """),format.raw/*162.13*/("""}"""),format.raw/*162.14*/("""


        """),format.raw/*165.9*/("""}"""),format.raw/*165.10*/(""") ;

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

            """),format.raw/*220.53*/("""
                """),format.raw/*221.96*/("""
                """),format.raw/*222.42*/("""
                    """),format.raw/*223.59*/("""
                    """),format.raw/*224.68*/("""
                """),format.raw/*225.27*/("""
            """),format.raw/*226.23*/("""
        """),format.raw/*227.9*/("""</div>
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
                  DATE: Tue Jan 22 15:42:20 AEDT 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/tsne/Tsne.scala.html
                  HASH: 51c7e6d5173b35cea086644cf8d6fa2f9b1e379a
                  MATRIX: 634->0|1612->1012|1649->1022|1988->1393|2025->1403|2659->2009|2688->2010|2730->2024|2852->2119|2881->2120|2920->2132|2955->2139|2984->2140|3026->2154|3213->2314|3242->2315|3281->2327|3319->2337|3348->2338|3390->2352|3512->2447|3541->2448|3580->2460|3613->2465|3642->2466|3684->2480|3842->2611|3871->2612|3910->2624|3944->2630|3973->2631|4015->2645|4130->2733|4159->2734|4198->2746|4233->2753|4262->2754|4304->2768|4358->2795|4387->2796|4426->2808|4457->2811|4486->2812|4528->2826|4618->2889|4647->2890|4686->2902|4718->2906|4747->2907|4789->2921|4951->3056|4980->3057|5020->3069|5055->3075|5085->3076|5128->3090|5178->3112|5208->3113|5248->3125|5282->3130|5312->3131|5355->3145|5477->3239|5507->3240|5545->3250|5642->3318|5672->3319|5715->3333|5783->3372|5813->3373|5860->3391|6146->3648|6176->3649|6245->3689|6275->3690|6322->3708|6395->3752|6425->3753|6476->3775|6644->3914|6674->3915|6729->3941|7025->4208|7055->4209|7114->4239|7202->4298|7232->4299|7283->4321|7313->4322|7366->4346|7535->4486|7565->4487|7618->4511|7648->4512|7699->4534|7796->4602|7826->4603|7869->4617|7899->4618|7978->4668|8008->4669|8055->4687|8091->4694|8121->4695|8172->4717|8338->4854|8368->4855|8408->4866|8438->4867|8570->4970|8600->4971|8655->4997|8772->5085|8802->5086|8903->5158|8933->5159|8988->5185|9160->5328|9190->5329|9237->5347|9267->5348|9312->5364|9342->5365|9384->5379|9414->5380|11519->7496|11566->7593|11613->7636|11664->7696|11715->7765|11762->7793|11805->7817|11843->7827
                  LINES: 25->1|48->24|49->25|55->31|56->32|75->51|75->51|76->52|79->55|79->55|81->57|81->57|81->57|82->58|87->63|87->63|89->65|89->65|89->65|90->66|93->69|93->69|95->71|95->71|95->71|96->72|100->76|100->76|102->78|102->78|102->78|103->79|106->82|106->82|108->84|108->84|108->84|109->85|110->86|110->86|112->88|112->88|112->88|113->89|115->91|115->91|117->93|117->93|117->93|118->94|122->98|122->98|124->100|124->100|124->100|125->101|126->102|126->102|128->104|128->104|128->104|129->105|132->108|132->108|133->109|136->112|136->112|137->113|137->113|137->113|138->114|144->120|144->120|146->122|146->122|147->123|147->123|147->123|148->124|150->126|150->126|151->127|153->129|153->129|154->130|155->131|155->131|156->132|156->132|158->134|162->138|162->138|162->138|162->138|163->139|165->141|165->141|166->142|166->142|169->145|169->145|170->146|170->146|170->146|171->147|174->150|174->150|174->150|174->150|176->152|176->152|177->153|179->155|179->155|180->156|180->156|181->157|184->160|184->160|185->161|185->161|186->162|186->162|189->165|189->165|244->220|245->221|246->222|247->223|248->224|249->225|250->226|251->227
                  -- GENERATED --
              */
          