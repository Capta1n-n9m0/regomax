const fs = require("fs").promises;
const path = require("path");
const bsplit = require("buffer-split");


function processFilename(filename){
    return path.join(__dirname, "..", "Data", filename);
}

class Network{
    async constructor(filename) {
        console.log(processFilename(filename));
        await this.read_network(filename);
    }
    async read_network(filename){
        const data = await fs.readFile(processFilename(filename));
        let lines = bsplit(data, Buffer.from("\n"));
        this.size = parseInt(lines[0].toString());
        this.link_len = parseInt(lines[1].toString());
        this.init_mem();
        this.base_name = filename.split(".")[0];
        console.log(this.base_name);
        lines = lines.slice(2);
        for(let i = 0; i < lines.length; i++){
            const line = lines[i];
            if(line.length === 0) continue;
            const numbers = bsplit(line, Buffer.from(" "));
            // no buffers for now
            // const n1 = BigInt(numbers[0].toString());
            // const n2 = BigInt(numbers[1].toString());
            // this.from.writeBigInt64LE(n1, i*8);
            // this.from.writeBigInt64LE(n2, i*8);
            const n1 = parseInt(numbers[0].toString());
            const n2 = parseInt(numbers[1].toString());
            this.from.push(n1-1);
            this.to.push(n2-1);
        }
        this.complete();
    }
    init_mem(){
        // i don't to want to bother with buffers yet.
        // so, using js arrays
        // this.from = Buffer.alloc(8*this.link_len);
        // this.to = Buffer.alloc(8*this.link_len);
        // this.link_num = Buffer.alloc(8*this.size);
        // this.firstpos = Buffer.alloc(8*this.size);
        this.from = [];
        this.to = [];
        this.link_num = [];
        this.firstpos = [];
    }
    complete(){
        let i, jj;
        let dangling_len;
        jj = 0;
        this.firstpos[jj] = 0;
        for(i = 0; i < this.link_len; i++){
            while (jj < from[i]){
                jj++;
                this.firstpos[jj] = i;
            }
        }
        while (jj < this.size){
            jj++;
            this.firstpos[jj] = i;
        }
        dangling_len = 0;
        for(i = 0; i < this.size; i++){
            if(this.firstpos[i] === this.firstpos[i+1]){
                dangling_len++;
            }
        }
        this.dangling = [];
        if(dangling_len > 0){
            for(i = 0; i < this.size; i++){
                if(this.firstpos[i] === this.firstpos[i+1]){
                    this.dangling.push(i);
                }
            }
        }
        for(jj = 0; jj < this.size; jj++){
            this.link_num[jj] = this.firstpos[jj+1] - this.firstpos[jj];
        }
    }
}

module.exports = Network;