const base_url = "http://localhost:3000"

const chartLoader = document.getElementById('chart-loading-spinner');
const chartCanvas = document.getElementById('lrChart');
const detailPanel = document.getElementById('detailPanel');

function set_loading(loading) {
    if(loading === true){
        chartCanvas.style.display = "none";
        chartLoader.style.display = "block";
        detailPanel.disabled = true;
    }
    else {
        chartLoader.style.display = "none";
        chartCanvas.style.display = "block";
        detailPanel.disabled = false;
    }
}

async function weather_lstm() {
    //Lesen der Argumente
    const days_to_forecast = document.getElementById("forecast_days").value  
    const lstm_epochs = document.getElementById("lstm_epochs").value

    if(days_to_forecast != null  || lstm_epochs != null ){
        const request = {"dates":"2021-01-10\n2021-01-11", "days": days_to_forecast, "epochs": lstm_epochs}
        const url = base_url + "/lstm"
        

        console.log("makeing request to " + url)
        try {
            set_loading(true)
            const response = await fetch(url, {
                method: 'POST', // *GET, POST, PUT, DELETE, etc.
                //mode: 'cors', // no-cors, *cors, same-origin
                //cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
                //credentials: 'same-origin', // include, *same-origin, omit
                headers: {
                'Content-Type': 'application/json',
                //'Access-Control-Allow-Origin': '*',
                'dates':'2021-01-10 2021-01-11',
                'days': '${days_to_forecast}',
                'epochs': '${lstm_epochs}'
                // 'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: JSON.stringify(request) // body data type must match "Content-Type" header
            });
            set_loading(false)
            console.log(response.status)
            return response.json(); // parses JSON response into native JavaScript objects
        } catch (error) {
            set_loading(false)
            console.log(error)
        }
    }
    else {
        return;
    }

}