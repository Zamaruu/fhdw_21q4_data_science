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