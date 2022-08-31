const fs = require("fs");
const path = require("path");
const bsplit = require("buffer-split");


const network_filename = process.argv[2];
const delta_alpha = process.argv[3];
const iprint = process.argv[4];
const print_number = process.argv[5];
const ten_number = process.argv[6];
const selected_filename = process.argv[7];
const names_filename = process.argv[8];
console.log(network_filename)
console.log(delta_alpha)
console.log(iprint)
console.log(print_number)
console.log(ten_number)
console.log(selected_filename)
console.log(names_filename)

let network_links, selected_nodes, selected_names;

fs.readFile(path.join(__dirname, network_filename), (err, data)=>{
    if(err) {
        console.log(err);
        throw err;
    }
    const lines = bsplit(data, Buffer.from("\n"));
    console.log(lines[0].toString());
    console.log(lines.length);
});
fs.readFile(path.join(__dirname, selected_filename), (err, data)=>{
    if(err){
        console.log(err);
        throw err;
    }
    const lines = bsplit(data, Buffer.from("\n"));
    console.log(lines[0].toString());
    console.log(lines.length);
});
if(names_filename) {
    fs.readFile(path.join(__dirname, names_filename), (err, data)=>{
        if(err){
            console.log(err);
            throw err;
        }
        const lines = bsplit(data, Buffer.from("\n"));
        console.log(lines[0].toString());
        console.log(lines.length);
    })
}
