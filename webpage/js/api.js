const base_url = "http://localhost:3000"

const chartLoader = document.getElementById('chart-loading-spinner');
const chartCanvas = document.getElementById('lrChart');
const detailPanel = document.getElementById('detailPanel');
const train_test_chart = document.getElementById("knnPredChart");

function set_loading(loading, show_train_chart) {
    if(loading === true){
        chartCanvas.style.display = "none";
        chartLoader.style.display = "block";
        detailPanel.disabled = true;
        if(show_train_chart){
            train_test_chart.style.display = "none";
        }
    }
    else {
        chartLoader.style.display = "none";
        chartCanvas.style.display = "block";
        detailPanel.disabled = false;
        if(show_train_chart){
            train_test_chart.style.display = "block";
        }
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
            set_loading(true, true)
            const response = await fetch(url, {
                method: 'POST', // *GET, POST, PUT, DELETE, etc.
                headers: {
                'Content-Type': 'application/json',
                //'Access-Control-Allow-Origin': '*',
                },
                body: JSON.stringify(request) // body data type must match "Content-Type" header
            });
            set_loading(false, true)
            console.log(response.status)
            const data = await response.json()
            console.log(data); // parses JSON response into native JavaScript objects
            set_loading(false, true)
            lstm_forecast_chart(data["past_date"], data["past_tavg"], data["forecast_dates"], data["forecast_tavg"]);
            lstm_training_chart(data["train_date"], data["train_tavg"], data["valid_date"], data["valid_tavg"], data["prediction_date"], data["prediction_tavg"],)
            loss_chart(data["loss_history"])
            mae_chart(data["mae_history"])
            document.getElementById("days").textContent= "Forecast-Tage: " + data["days"];
            document.getElementById("epochs").textContent= "Epochen: "+ data["epochs"];
            document.getElementById("rmse").textContent= "RMSE: " + (data["rmse"]).toFixed(2);
            document.getElementById("runtime").textContent= "Runtime: " + Number((data["runtime"]).toFixed(2)) + " Sekunden";
        } catch (error) {
            set_loading(false, true)
            console.log(error)
        }
    }
    else {
        return;
    }

}

async function weather_lr() {
    //Lesen der Argumente
    const start = document.getElementById("forecast_start").value  
    const end = document.getElementById("forecast_end").value
    
    if(start != null  || end != null ){
        const request = {"start": start, "end": end}
        const url = base_url + "/tf"        

        console.log("makeing request to " + url)
        try {
            set_loading(true, false)
            const response = await fetch(url, {
                method: 'POST', // *GET, POST, PUT, DELETE, etc.
                headers: {
                'Content-Type': 'application/json',
                //'Access-Control-Allow-Origin': '*',
                },
                body: JSON.stringify(request) // body data type must match "Content-Type" header
            });
            set_loading(false, false)
            console.log(response.status)
            const data = await response.json()
            console.log(data); // parses JSON response into native JavaScript objects
            set_loading(false, false)
            document.getElementById('detailPanel').disabled = true;;
            lr_forecast(data["past_date"], data["past_tavg"], data["forecast_dates"], data["forecast_tavg"], data["forecast_linear"]);
            // loss_chart(data["loss_history"])
            // mae_chart(data["mae_history"])
            // document.getElementById("days").textContent= "Forecast-Tage: " + data["days"];
            // document.getElementById("epochs").textContent= "Epochen: "+ data["epochs"];
            // document.getElementById("rmse").textContent= "RMSE: " + data["rmse"];
        } catch (error) {
            set_loading(false, false)
            console.log(error)
        }
    }
    else {
        return;
    }

}

async function weather_acf() {
    //Lesen der Argumente
    const start = document.getElementById("acf_start").value  
    const end = document.getElementById("acf_end").value
    
    if(start != null  || end != null ){
        const request = {"start": start, "end": end}
        const url = base_url + "/acf"        

        console.log("makeing request to " + url)
        try {
            set_loading(true, false)
            const response = await fetch(url, {
                method: 'POST', // *GET, POST, PUT, DELETE, etc.
                headers: {
                'Content-Type': 'application/json',
                //'Access-Control-Allow-Origin': '*',
                },
                body: JSON.stringify(request) // body data type must match "Content-Type" header
            });
            set_loading(false, false)
            console.log(response.status)
            const data = await response.json()
            console.log(data); // parses JSON response into native JavaScript objects
            document.getElementById('detailPanel').disabled = true;
            acf_chart(data["acf"], data["ci_pos"], data["ci_neg"]);
            // loss_chart(data["loss_history"])
            // mae_chart(data["mae_history"])
            // document.getElementById("days").textContent= "Forecast-Tage: " + data["days"];
            // document.getElementById("epochs").textContent= "Epochen: "+ data["epochs"];
            // document.getElementById("rmse").textContent= "RMSE: " + data["rmse"];
        } catch (error) {
            set_loading(false, false)
            console.log(error)
        }
    }
    else {
        return;
    }

}


// Chart Functions
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

function lr_forecast(old_dates, old_tavg, forecast_dates, forecast_tavg, forecast_linear) {
    for(var i = 0; i < old_dates.length; i++){
        forecast_tavg.unshift(null);
    }

    for(var i = 0; i < old_dates.length; i++){
        forecast_linear.unshift(null);
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
                    label: 'Forecast mit DNN',
                    data: forecast_tavg,
                    borderColor: 'rgb(204, 24, 24)',
                    tension: 0.1
                },
                {
                    type: 'line',
                    label: 'Forecast mit LR',
                    data: forecast_linear,
                    borderColor: 'rgb(204, 0, 0)',
                    tension: 0.1
                },
        ],
        },
    });
}

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

function lstm_training_chart(train_dates, train_tavg, test_dates, test_tavg, pred_dates, pred_tavg){
    for(var i = 0; i < train_dates.length; i++){
        test_tavg.unshift(null);
        pred_tavg.unshift(null);
    }

    const ctx = document.getElementById('knnPredChart');
    train_test_chart.style.display = "block";
    const myChart = new Chart(ctx, {
        data: {
            labels: train_dates.concat(test_dates),
            datasets: [
                {
                    type: 'line',
                    label: 'Trainings Daten',
                    data: train_tavg,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                },
                {
                    type: 'line',
                    label: 'Prediction Daten',
                    data: pred_tavg,
                    borderColor: 'rgb(204, 24, 24)',
                    tension: 0.1
                },
                {
                    type: 'line',
                    label: 'Validierungsdaten',
                    data: test_tavg,
                    borderColor: 'rgb(24, 204, 24)',
                    tension: 0.1
                },
        ],
        },
    });
}

function acf_chart(acf, ci_pos, ci_neg){
    // for(var i = 0; i < train_dates.length; i++){
    //     test_tavg.unshift(null);
    //     pred_tavg.unshift(null);
    // }

    x_values = Array.from(Array(acf.length).keys())

    const ctx = document.getElementById('lrChart');
    const myChart = new Chart(ctx, {
        data: {
            labels: x_values,
            datasets: [
                {
                    type: 'line',
                    label: 'ci_pos Daten',
                    data: ci_pos,
                    borderColor: 'rgb(204, 24, 24)',
                    tension: 0.1
                },
                {
                    type: 'line',
                    label: 'ci_neg Daten',
                    data: ci_neg,
                    borderColor: 'rgb(204, 24, 24)',
                    tension: 0.1
                },
                {
                    type: 'line',
                    label: 'ACF',
                    data: acf,
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                },
        ],
        },
    });
}