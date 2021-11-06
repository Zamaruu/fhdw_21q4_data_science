function startIndexScript(){
    var chartLoader = document.getElementById('chart-loading-spinner');
    var chartCanvas = document.getElementById('lrChart');
    var detailPanel = document.getElementById('detailPanel');

    chartCanvas.style.display = "none";
    chartLoader.style.display = "block";
    detailPanel.disabled = true;

    setTimeout(function() {
        // Code, der erst nach 2 Sekunden ausgeführt wird
        chartLoader.style.display = "none";
        chartCanvas.style.display = "block";
        detailPanel.disabled = false;
    }, 2570);
}