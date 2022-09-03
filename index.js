process.env.UV_THREADPOOL_SIZE = 6;

const App = require("./Source/App");

App(process.argv);