const Vector = require("./Vector");
const fs = require("fs");
const bsplit = require("buffer-split");
const printf = require("printf");
const path = require("path");

/**
 * @param {string} filename
 * @param {string} folder
 * @return string
 */
function processFilename(filename, folder = "Data") {
    return path.join(__dirname, "..", folder, filename);
}

class Matrix{
    /** @type {number}
     */
    ydim;
    /** @type {number}
     */
    xdim;
    /** @type {Vector<Vector>}
     */
    mat;

    /**
     * @param{number | Matrix} x
     * @param{number} y
     * @param{number} diag
     */
    constructor(x, y, diag = 0) {
        if(x instanceof Matrix){
            this.xdim = x.xdim; this.ydim = x.ydim;
            this.init_mem();
            for(let i = 0; i < this.ydim; i++) this.mat[i].eq(x.mat[i]);
        } else {
            let i;
            this.xdim = x;
            this.ydim = y;
            this.init_mem();
            for (i = 0; i < this.ydim; i++) {
                this.mat[i].put_value(0);
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
        this.mat = new Vector(this.ydim);
        for(let i = 0; i < this.ydim; i++){
            this.mat[i] = new Vector(this.xdim);
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
        for(let i = 0; i < this.ydim; i++) this.mat[i].eq(a.mat[i]);
    }

    /**
     * @param {Matrix} a
     * @param {string} filename
     * @param {string} node_file_names
     */
    static print_mat(a, filename, node_file_names = null) {
        let i, j, dimx, dimy, len = 0, nlen, l;
        let node_names = [];
        if (node_file_names !== null) {
            const data = fs.readFileSync(processFilename(node_file_names));
            const lines = bsplit(data, Buffer.from("\n"));
            for (const line of lines) {
                if (line.length === 0) continue;
                node_names.push(line.toString());
            }
        }
        dimx = a.x;
        dimy = a.y;
        let fp = fs.openSync(processFilename(filename, "Results"), 'w');
        let buffer = "";
        for (let i = 0; i < dimy; i++) {
            for (let j = 0; j < dimx; j++) {
                buffer += printf("%5d\t  %5d\t  %24.26lg", i, j, a.mat[i][j]);
                if (i < len && j < len) {
                    buffer += printf("\t%s\t", node_names[i], node_file_names[j]);
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