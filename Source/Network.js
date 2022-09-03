const fs = require("fs");
const path = require("path");
const bsplit = require("buffer-split");
const {format} = require("util");


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
    /**
     * @param {string} filename
     */
    constructor(filename) {
        if (filename) {
            this.read_network(filename);
            console.log(processFilename(filename));
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
            // no buffers for now
            // const n1 = BigInt(numbers[0].toString());
            // const n2 = BigInt(numbers[1].toString());
            // this.from.writeBigInt64LE(n1, i*8);
            // this.from.writeBigInt64LE(n2, i*8);
            const n1 = parseInt(numbers[0].toString());
            const n2 = parseInt(numbers[1].toString());
            this.from.push(n1 - 1);
            this.to.push(n2 - 1);
        }
        this.complete();
        console.log("size = %d   link_len = %d   dangling_len = %d", this.size, this.link_len, this.dangling.length);
        console.log("****** => Reading of data file finished");
    }

    init_mem() {
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

    complete() {
        let i, jj;
        let dangling_len;
        jj = 0;
        this.firstpos[jj] = 0;
        for (i = 0; i < this.link_len; i++) {
            while (jj < this.from[i]) {
                jj++;
                this.firstpos[jj] = i;
            }
        }
        while (jj < this.size) {
            jj++;
            this.firstpos[jj] = i;
        }
        dangling_len = 0;
        for (i = 0; i < this.size; i++) {
            if (this.firstpos[i] === this.firstpos[i + 1]) {
                dangling_len++;
            }
        }
        this.dangling = [];
        for (i = 0; i < dangling_len; i++) this.dangling.push(0);
        if (dangling_len > 0) {
            dangling_len = 0;
            for (i = 0; i < this.size; i++) {
                if (this.firstpos[i] === this.firstpos[i + 1]) {
                    this.dangling[dangling_len++] = i;
                }
            }
        }
        for (jj = 0; jj < this.size; jj++) {
            this.link_num[jj] = this.firstpos[jj + 1] - this.firstpos[jj];
        }
    }

    /**
     * @param {float} delta_alpha
     * @param {float[]} output
     * @param {float[]} input
     * @param {number} norm_flag
     */
    GTmult(delta_alpha, output, input, norm_flag = 1) {
        let sum, val;
        let i, a, b, jj;

        // contribution from dangling modes
        // note the modification with respect to GGmult
        // ==> 1/N d e^T
        if (this.dangling.length > 0) {
            sum = 0.0;
            for (i = 0; i < this.size; i++) {
                output[i] = 0;
                sum += input[i];
            }
            sum /= this.size;
            for (i = 0; i < this.dangling.length; i++) output[this.dangling[i]] += sum;
        } else {
            for (i = 0; i < this.size; i++) {
                output[i] = 0;
            }
        }

        //  Computation of out=S^T*in
        for (jj = 0; jj < this.size; jj++) {
            a = this.firstpos[jj];
            b = this.firstpos[jj + 1];
            if (a >= b) continue;
            // note that from[a]=from[i]=jj for a<=i<b
// #ifndef USE_PROBS
            sum = 0;
            for (i = a; i < b; i++) sum += input[this.to[i]];
            output[jj] += sum / this.link_num[jj];
// #else
            //    for(i=a;i<b;i++) sum+=prob[i]*in[to[i]];
            //    out[jj]+=sum;
            //    for(i=a;i<b;i++) out[jj]+=prob[i]*in[to[i]];
// #endif

            if (delta_alpha === 0) return;
            val = 1.0 - delta_alpha;
            for (i = 0; i < this.size; i++) output[i] *= val;
            if (norm_flag) {
                sum = 1;
            } else {
                sum = 0;
                for (i = 0; i < this.size; i++) sum += input[i];
            }
            sum *= delta_alpha / this.size;
            for (i = 0; i < this.size; i++) output[i] += sum;
        }
    }

    /**
     * @param {float} delta_alpha
     * @param {float[]} output
     * @param {float[]} input
     * @param {number} norm_flag
     */
    GGmult(delta_alpha, output, input, norm_flag = 1) {
        let sum, val;
        let i, a, b, jj;

        // contribution from dangling modes
        // ==> 1/N e d^T
        sum = 0.0;
        if (this.dangling.dim > 0) {
            for (i = 0; i < this.dangling.length; i++) {
                sum += input[this.dangling[i]];
            }
            sum /= this.size;
        }
        for (i = 0; i < this.size; i++) output[i] = sum;


        //  Computation of out=S*in
        for (jj = 0; jj < this.size; jj++) {
            a = this.firstpos[jj];
            b = this.firstpos[jj + 1];
            if (a >= b) continue;

// #ifndef USE_PROBS
            //    val=in[from[a]]/(b-a);
            //    val=in[jj]/(b-a);
            val = input[jj] / this.link_num[jj];
// #else
            //    val=in[from[a]];
            // val=in[jj];
// #endif
            // val = in[from[a]]*prob[a]
            for (i = a; i < b; i++) {
// #ifndef USE_PROBS
                output[this.to[i]] += val;
// #else
                // out[to[i]]+=prob[i]*val;
// #endif
            }
        }

        // computation of out=G*in, i.e. damping factor contributions
        // avoid comlications and rounding errors if delta_alpha==0
        if (delta_alpha === 0) return;
        val = 1.0 - delta_alpha;
        for (i = 0; i < this.size; i++) output[i] *= val;
        if (norm_flag) {
            sum = 1;
        } else {
            sum = 0;
            for (i = 0; i < this.size; i++) sum += input[i];
        }
        sum *= delta_alpha / this.size;
        for (i = 0; i < this.size; i++) output[i] += sum;
    }

}

module.exports = Network;