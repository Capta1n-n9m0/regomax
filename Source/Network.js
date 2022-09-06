const fs = require("fs");
const path = require("path");
const bsplit = require("buffer-split");
const Vector = require("./Vector");


/**
 * @param {string} filename
 * @param {string} folder
 * @return string
 */
function processFilename(filename, folder = "Data") {
    console.log(filename);
    return path.join(__dirname, "..", folder, filename);
}

class Network {

    /** @type {number}
     */
    size;
    /** @type {number}
     */
    link_len;
    /** @type {string}
     */
    base_name;
    /** @type {Vector}
     */
    from = new Vector();
    /** @type {Vector}
     */
    to = new Vector();
    /** @type {Vector}
     */
    link_num = new Vector();
    /** @type {Vector}
     */
    firstpos = new Vector();
    /** @type {Vector}
     */
    dangling = new Vector();

    /**
     * @param {Network} o
     * @returns {Network}
     */
    static fromObj(o){
        const res = new Network();

        res.size = o.size;
        res.link_len = o.size;
        res.base_name = o.base_name;
        res.from = Vector.fromObj(o.from);
        res.to = Vector.fromObj(o.to);
        res.link_num = Vector.fromObj(o.link_num);
        res.firstpos = Vector.fromObj(o.firstpos);
        res.dangling = Vector.fromObj(o.dangling);

        return res;
    }
    /**
     * @param {string} filename
     */
    constructor(filename=undefined) {
        if (filename) {
            this.read_network(filename);
        }
    }

    /**
     * @param {string} filename
     */
    read_network(filename) {
        console.log("\n****** => Reading of data file ");
        const data = fs.readFileSync(processFilename(filename));
        let lines = bsplit(data, Buffer.from("\n"));
        this.size = parseInt(lines[0].toString());
        this.link_len = parseInt(lines[1].toString());
        this.init_mem();
        this.base_name = filename.split(".")[0];
        console.log("****** => Reading of integer connection matrix");
        lines = lines.slice(2);
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            if (line.length === 0) continue;
            const numbers = bsplit(line, Buffer.from(" "));
            const n1 = parseInt(numbers[0].toString());
            const n2 = parseInt(numbers[1].toString());
            this.from.c[i] = n1 - 1;
            this.to.c[i] = n2 - 1;
        }
        this.complete();
        console.log("size = %d   link_len = %d   dangling_len = %d", this.size, this.link_len, this.dangling.dim);
        console.log("****** => Reading of data file finished");
    }

    init_mem() {
        // I don't to want to bother with buffers yet.
        // so, using js arrays
        // this.from = Buffer.alloc(8*this.link_len);
        // this.to = Buffer.alloc(8*this.link_len);
        // this.link_num = Buffer.alloc(8*this.size);
        // this.firstpos = Buffer.alloc(8*this.size);
        this.from = new Vector(this.link_len);
        this.to = new Vector(this.link_len);
        this.link_num = new Vector(this.size);
        this.firstpos = new Vector(this.size + 1);
    }

    complete() {
        let i, jj;
        let dangling_len;

        // building of 1st pos-structure
        jj = 0;
        this.firstpos.c[jj] = 0;
        for (i = 0; i < this.link_len; i++) {
            while (jj < this.from.c[i]) {
                jj++;
                this.firstpos.c[jj] = i;
            }
        }
        while (jj < this.size) {
            jj++;
            this.firstpos.c[jj] = i;
        }

        // calculation of dangling nodes
        dangling_len = 0;
        for (i = 0; i < this.size; i++) {
            if (this.firstpos.c[i] === this.firstpos.c[i + 1]) {
                dangling_len++;
            }
        }
        this.dangling.resize(dangling_len);

        if (dangling_len > 0) {
            dangling_len = 0;
            for (i = 0; i < this.size; i++) {
                if (this.firstpos.c[i] === this.firstpos.c[i + 1]) {
                    this.dangling.c[dangling_len++] = i;
                }
            }
        }

        // compute link_num vector
        for (jj = 0; jj < this.size; jj++) {
            this.link_num.c[jj] = this.firstpos.c[jj + 1] - this.firstpos.c[jj];
        }
    }

    /**
     * @param {number} delta_alpha
     * @param {Vector} output
     * @param {Vector} input
     * @param {number} norm_flag
     */
    GTmult(delta_alpha, output, input, norm_flag = 1) {
        let sum, val;
        let i, a, b, jj;

        // contribution from dangling modes
        // note the modification with respect to GGmult
        // ==> 1/N d e^T
        if (this.dangling.dim > 0) {
            sum = 0.0;
            for (i = 0; i < this.size; i++) {
                output.c[i] = 0;
                sum += input.c[i];
            }
            sum /= this.size;
            for (i = 0; i < this.dangling.dim; i++) output.c[this.dangling.c[i]] += sum;
        } else {
            for (i = 0; i < this.size; i++) {
                output.c[i] = 0;
            }
        }

        //  Computation of out=S^T*in
        for (jj = 0; jj < this.size; jj++) {
            a = this.firstpos.c[jj];
            b = this.firstpos.c[jj + 1];
            if (a >= b) continue;
            // note that from[a]=from[i]=jj for a<=i<b
// #ifndef USE_PROBS
            sum = 0;
            for (i = a; i < b; i++) sum += input.c[this.to.c[i]];
            output.c[jj] += sum / this.link_num.c[jj];
// #else
            //    for(i=a;i<b;i++) sum+=prob[i]*in[to[i]];
            //    out[jj]+=sum;
            //    for(i=a;i<b;i++) out[jj]+=prob[i]*in[to[i]];
// #endif
        }

        // computation of out=G^T*in, i.e. damping factor contributions
        // avoid complications and rounding errors if delta_alpha==0
        if (delta_alpha === 0) return;
        val = 1.0 - delta_alpha;
        for (i = 0; i < this.size; i++) output.c[i] *= val;
        if (norm_flag) {
            sum = 1;
        } else {
            sum = 0;
            for (i = 0; i < this.size; i++) sum += input.c[i];
        }
        sum *= delta_alpha / this.size;
        for (i = 0; i < this.size; i++) output.c[i] += sum;
    }

    /**
     * @param {number} delta_alpha
     * @param {Vector} output
     * @param {Vector} input
     * @param {number} norm_flag
     */
    GGmult(delta_alpha, output, input, norm_flag = 1) {
        let sum, val;
        let i, a, b, jj;

        // contribution from dangling modes
        // ==> 1/N e d^T
        sum = 0.0;
        if (this.dangling.dim > 0) {
            for (i = 0; i < this.dangling.dim; i++) {
                sum += input.c[this.dangling.c[i]];
            }
            sum /= this.size;
        }
        for (i = 0; i < this.size; i++) output.c[i] = sum;


        //  Computation of out=S*in
        for (jj = 0; jj < this.size; jj++) {
            a = this.firstpos.c[jj];
            b = this.firstpos.c[jj + 1];
            if (a >= b) continue;
// #ifndef USE_PROBS
            //    val=in[from[a]]/(b-a);
            //    val=in[jj]/(b-a);
            val = input.c[jj] / this.link_num.c[jj];
// #else
            //    val=in[from[a]];
            // val=in[jj];
// #endif
            // val = in[from[a]]*prob[a]
            for (i = a; i < b; i++) {
// #ifndef USE_PROBS
                output.c[this.to.c[i]] += val;
// #else
                // out[to[i]]+=prob[i]*val;
// #endif
            }
        }

        // computation of out=G*in, i.e. damping factor contributions
        // avoid complications and rounding errors if delta_alpha==0
        if (delta_alpha === 0) return;
        val = 1.0 - delta_alpha;
        for (i = 0; i < this.size; i++) output.c[i] *= val;
        if (norm_flag) {
            sum = 1;
        } else {
            sum = 0;
            for (i = 0; i < this.size; i++) sum += input.c[i];
        }
        sum *= (delta_alpha / this.size);
        for (i = 0; i < this.size; i++) output.c[i] += sum;
    }

}

module.exports = Network;