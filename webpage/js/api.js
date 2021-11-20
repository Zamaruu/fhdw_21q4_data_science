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
                headers: {
                'Content-Type': 'application/json',
                //'Access-Control-Allow-Origin': '*',
                'dates':'2021-01-10 2021-01-11',
                'days': '${days_to_forecast}',
                'epochs': '${lstm_epochs}'
                },
                body: JSON.stringify(request) // body data type must match "Content-Type" header
            });
            set_loading(false)
            console.log(response.status)
            const data = await response.json()
            console.log(data); // parses JSON response into native JavaScript objects
            set_loading(false)
            lstm_forecast_chart(data["past_date"], data["past_tavg"], data["forecast_dates"], data["forecast_tavg"]);
            loss_chart(data["loss_history"])
            mae_chart(data["mae_history"])
            document.getElementById("days").textContent= "Forecast-Tage: " + data["days"];
            document.getElementById("epochs").textContent= "Epochen: "+ data["epochs"];
            document.getElementById("rmse").textContent= "RMSE: " + data["rmse"];
        } catch (error) {
            set_loading(false)
            console.log(error)
        }
    }
    else {
        return;
    }

}


function mae_chart(mae){
    const ctx = document.getElementById('maeChart');
    const myChart = new Chart(ctx, {
        data: {
            labels: Array.from(Array(mae.length).keys()),
            datasets: [
                {
                    type: 'line',
                    label: 'MAE',
                    data: mae,
                    borderColor: 'rgb(18, 199, 84)',
                    tension: 0.1
                },
        ],
        },
    });
}

function loss_chart(loss){
    const ctx = document.getElementById('lossChart');
    const myChart = new Chart(ctx, {
        data: {
            labels: Array.from(Array(loss.length).keys()),
            datasets: [
                {
                    type: 'line',
                    label: 'LOSS',
                    data: loss,
                    borderColor: 'rgb(120, 18, 199)',
                    tension: 0.1
                },
        ],
        },
    });
}

// Chart Functions
function lstm_forecast_chart(old_dates, old_tavg, forecast_dates, forecast_tavg) {
    for(var i = 0; i < old_dates.length; i++){
        forecast_tavg.unshift(null);
    }

    const ctx = document.getElementById('lrChart');
    const myChart = new Chart(ctx, {
        data: {
            labels: old_dates.concat(forecast_dates),
            datasets: [
                {
                    type: 'line',
                    label: 'Vergangenheit',
                    data: old_tavg,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                },
                {
                    type: 'line',
                    label: 'Forecast mit LSTM',
                    data: forecast_tavg,
                    borderColor: 'rgb(204, 24, 24)',
                    tension: 0.1
                },
        ],
        },
    });
}