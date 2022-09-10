const fs = require("fs");
const bsplit = require("buffer-split");
const printf = require("printf");
const path = require("path");


class Matrix{
    /** @type {number}
     */
    ydim;
    /** @type {number}
     */
    xdim;
    /** @type {Float64Array[]}
     */
    mat;

    static fromObj(o){
        const res = new Matrix(1, 1);

        res.xdim = o.xdim;
        res.ydim = o.ydim;
        res.mat = o.mat;

        return res;
    }

    /**
     * @param{number | Matrix} x
     * @param{number} y
     * @param{number} diag
     */
    constructor(x, y, diag = 0) {
        if(x instanceof Matrix){
            this.xdim = x.xdim; this.ydim = x.ydim;
            this.init_mem();
            for (let i = 0; i < this.ydim; i++) {
                for (let j = 0; j < this.xdim; j++) {
                    this.mat[i][j] = x.mat[i][j];
                }
            }
        } else {
            let i;
            this.xdim = x;
            this.ydim = y;
            this.init_mem();
            for (i = 0; i < this.ydim; i++) {
                this.mat[i].fill(0);
            }
            if (diag !== 0) {
                const n = x < y ? x : y;
                for (i = 0; i < n; i++) {
                    this.mat[i][i] = diag;
                }
            }
        }
    }
    init_mem(){
        this.mat = new Array(this.ydim);
        for(let i = 0; i < this.ydim; i++){
            this.mat[i] = new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * this.xdim));
        }
    }

    /**
     * @returns {number}
     */
    get x(){
        return this.xdim;
    }

    /**
     * @returns {number}
     */
    get y(){
        return this.ydim;
    }

    /**
     * @param {Matrix} a
     */
    copy(a){
        this.xdim = a.xdim; this.ydim = a.ydim;
        this.init_mem();
        for (let i = 0; i < this.ydim; i++) {
            for (let j = 0; j < this.xdim; j++) {
                this.mat[i][j] = a.mat[i][j];
            }
        }
    }

    /**
     * @param {Matrix} a
     * @param {string} filename
     * @param {string} node_file_names
     */
    static print_mat(a, filename, node_file_names = undefined) {
        let dimx, dimy;
        let node_names = [];
        if (node_file_names) {
            const data = fs.readFileSync(node_file_names);
            /**
             * @type {Buffer[]}
             */
            const lines = bsplit(data, Buffer.from("\n"));
            for (let line of lines) {
                if (line.length === 0) continue;
                if (line.toString()[line.length-1] === "\r") line = line.slice(0, line.length-1);
                node_names.push(line.toString());
            }
        }
        dimx = a.x;
        dimy = a.y;
        let fp = fs.openSync(path.join(__dirname,"..", "Results", filename), 'w');
        let buffer = "";
        let len = node_names.length;
        for (let i = 0; i < dimy; i++) {
            for (let j = 0; j < dimx; j++) {
                buffer += printf("%5d\t  %5d\t  %24.26e", i, j, a.mat[i][j]);
                if (i < len && j < len) {
                    buffer += `\t\t${node_names[i]}\t${node_names[j]}`;
                }
                buffer += "\n";
            }
            buffer += "\n";
            fs.writeSync(fp, buffer);
            buffer = "";
        }
        fs.writeSync(fp, buffer);
        fs.closeSync(fp);
    }
}

module.exports = Matrix;