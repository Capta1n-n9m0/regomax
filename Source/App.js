const fs = require("fs");
const path = require("path");
const bsplit = require("buffer-split");
const Network = require("./Network");

function main(argv){
    const network_filename = argv[2];
    const delta_alpha = argv[3];
    const iprint = argv[4];
    const print_number = argv[5];
    const ten_number = argv[6];
    const selected_filename = argv[7];
    const names_filename = argv[8];
    console.log(network_filename)
    console.log(delta_alpha)
    console.log(iprint)
    console.log(print_number)
    console.log(ten_number)
    console.log(selected_filename)
    console.log(names_filename)

    let network_links = {}, selected_nodes = {}, selected_names = {};
    const net = new Network(network_filename);
    if(0){
        fs.readFile(path.join(__dirname, network_filename), (err, data) => {
            // parsing the whole network
            // 1st line is number of nodes
            // 2nd line is number of lines after itself
            if (err) {
                console.log(err);
                throw err;
            }
            const lines = bsplit(data, Buffer.from("\n"));
            // n_nodes and n_links can be used in the future for further optimisation
            network_links.n_nodes = parseInt(lines[0].toString());
            network_links.n_lines = parseInt(lines[1].toString());
            network_links.links = [];
            for (const line of lines.slice(2)) {
                if (line.length === 0) continue;
                const numbers = bsplit(line, Buffer.from(" "));
                const n1 = parseInt(numbers[0].toString());
                const n2 = parseInt(numbers[1].toString());
                network_links.links.push({f: n1, s: n2});
            }
            console.log("network_links");
            console.log(network_links.n_lines);
            console.log(network_links.n_nodes);
            console.log(network_links.links.length);
        });
        fs.readFile(path.join(__dirname, selected_filename), (err, data) => {
            // parsing number of selected nodes
            // 1st line is number of selected nodes and number of lines after itself at the same time
            if (err) {
                console.log(err);
                throw err;
            }
            const lines = bsplit(data, Buffer.from("\n"));
            selected_nodes.n_nodes = parseInt(lines[0].toString());
            selected_nodes.nodes = [];
            for (const line of lines.slice(1)) {
                if (line.length === 0) continue;
                selected_nodes.nodes.push(parseInt(line.toString()));
            }
            console.log("selected_nodes");
            console.log(selected_nodes.n_nodes);
            console.log(selected_nodes.nodes.length);
        });
        if (names_filename) {
            fs.readFile(path.join(__dirname, names_filename), (err, data) => {
                // if file containing names of selected nodes has been passed, it will be used to make output look better
                // it should contain the same number of lines as selected_nodes
                if (err) {
                    console.log(err);
                    throw err;
                }
                const lines = bsplit(data, Buffer.from("\n"));
                selected_names = [];
                for (const line of lines) {
                    if (line.length === 0) continue;
                    selected_names.push(line.toString());
                }
                console.log("selected_names");
                console.log(selected_names.length);
            });
        } else selected_names = undefined;
    }
}

module.exports = main;