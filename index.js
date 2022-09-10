const App = require("./Source/App");
const App_par = require("./Source/App_par");


if(process.argv[2] === "lin"){
    App(process.argv.slice(1));
} else if(process.argv[2] === "par"){
    App_par(process.argv.slice(1));
}else {
    console.log("First arg should be lin of par");
}

