/**
 * @author Maximilian Ditz
 * @description Diese Datei initilaisiert alle im Frontend verwendeten FabricJS-Elemente
 * @link https://developer.microsoft.com/en-us/fabric-js/getstarted/getstarted
 */


var ButtonElements = document.querySelectorAll(".ms-Button");
    for (var i = 0; i < ButtonElements.length; i++) {
        new fabric['Button'](ButtonElements[i], function() {
        // Insert Event Here
    });
}

var PanelExamples = document.getElementsByClassName("ms-PanelExample");
  for (var i = 0; i < PanelExamples.length; i++) {
    (function() {
      var PanelExampleButton = PanelExamples[i].querySelector(".ms-Button");
      var PanelExamplePanel = PanelExamples[i].querySelector(".ms-Panel");
      PanelExampleButton.addEventListener("click", function(i) {
        new fabric['Panel'](PanelExamplePanel);
      });
    }());
  }

var CalloutExamples = document.querySelectorAll(".ms-CalloutExample");
  for (var i = 0; i < CalloutExamples.length; i++) {
    var Example = CalloutExamples[i];
    var ExampleButtonElement = Example.querySelector(".ms-CalloutExample-button .ms-Button");
    var CalloutElement = Example.querySelector(".ms-Callout");
    new fabric['Callout'](
      CalloutElement,
      ExampleButtonElement,
      "right"
    );
  }

var SpinnerElements = document.querySelectorAll(".ms-Spinner");
  for (var i = 0; i < SpinnerElements.length; i++) {
    new fabric['Spinner'](SpinnerElements[i]);
  }

(function() {
  var example = document.querySelector(".docs-DialogExample-default");
  var button = example.querySelector(".docs-DialogExample-button");
  var dialog = example.querySelector(".ms-Dialog");
  var label = example.querySelector(".docs-DialogExample-label")
  var checkBoxElements = example.querySelectorAll(".ms-CheckBox");
  var actionButtonElements = example.querySelectorAll(".ms-Dialog-action");
  var checkBoxComponents = [];
  var actionButtonComponents = [];
  // Wire up the dialog
  var dialogComponent = new fabric['Dialog'](dialog);
  // Wire up the checkBoxes
  for (var i = 0; i < checkBoxElements.length; i++) {
    checkBoxComponents[i] = new fabric['CheckBox'](checkBoxElements[i]);
  }
  // Wire up the buttons
  for (var i = 0; i < actionButtonElements.length; i++) {
    actionButtonComponents[i] = new fabric['Button'](actionButtonElements[i], actionHandler);
  }
  // When clicking the button, open the dialog
  button.onclick = function() {
    openDialog(dialog);
  };
  function actionHandler(event) {
    var labelText = "";
    var counter = 0;
    for (var i = 0; i < checkBoxComponents.length; i++) {
      if (checkBoxComponents[i].getValue()) {
        counter++;
      }
    }
  }
  function openDialog(dialog) {
    // Open the dialog
    dialogComponent.open();
  }
}());


var TextFieldElements = document.querySelectorAll(".ms-TextField");
  for (var i = 0; i < TextFieldElements.length; i++) {
    new fabric['TextField'](TextFieldElements[i]);
  }
