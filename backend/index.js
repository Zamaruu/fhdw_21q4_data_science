const express = require('express')
const app = express()
const port = 3000;
app.use(express.json())

var fs = require('fs')
var { PythonShell } = require('python-shell')

var apis = ["lstm", "tf", "acf", "test"];

app.get('/', (req, res) => {
    var string = "Apis können aufgerufen werden durch: url:3000/[name]\n\n";
    string += "Im Body sollte ein JSON sein, das unter dem key 'dates' eine durch '\\n' getrennte liste von datestrings beinhalten soll.\n"
    string += "bsp.: {'dates': '2021-01-10\\n2021-01-11'}\n\n"
    string += "Mögliche Apis = " + apis.join(', ')

    res.send(string)
})

for (var idx in apis) {
    var name = apis[idx]
    console.log("listen to path 'localhost:" + port + "/" + name + "'.")
    app.post("/" + name, async (req, res) => {
        try {
            await makeFile(req.body['dates'])
            await startPython(name)
            var response = await readPythonOutput()
            res.json({ 0: response })
        } catch (e) {
            returnError
        }

    })
}

/**
 * @description Handles error during an reponse
 * @param {response} res 
 */
var returnError = function (res, error) {
    res.status(500).send(error)
}

/**
 * @description Makes file for import in python
 */
var makeFile = function (content) {
    return new Promise((resolve, reject) => {
        fs.writeFile('import.csv', content, (e) => {
            if (e) {
                reject(e)
                return
            }
            resolve()
        })
    })
}

/**
 * @description Starts Python file by name
 * @param {String} name 
 */
var startPython = function (name) {
    return new Promise((resolve, reject) => {
        PythonShell.run('../python/weather_' + name + '.py', null, (e) => {
            if (e) {
                reject(e)
                return
            }
            resolve()
        })
    })
}

/**
 * @description returns Output from Python (per file)
 */
var readPythonOutput = function () {
    return new Promise((resolve, reject) => {
        fs.readFile('output.csv', "utf8", (e, data) => {
            if (e) {
                reject(e)
                return
            }
            resolve(data)
        })
    })
}

app.listen(port, () => {
    console.log(`App hört auf Port: ${port}`)
})