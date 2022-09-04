const Vector = require("./Vector");

class Matrix{
    /**
     * @param{number} x
     * @param{number} y
     * @param{number} diag
     */
    constructor(x, y, diag = 0) {
        let i;
        this.xdim = x;
        this.ydim = y;
        this.init_mem();
        for(i = 0; i < this.ydim; i++) {
            this.mat[i].fill(0);
        }
        if(diag !== 0){
            const n = x<y ? x : y;
            for(i = 0; i < n; i++){
                this.mat[i][i] = diag;
            }
        }
    }
    init_mem(){
        this.mat = new Array(this.ydim);
        for(let i = 0; i < this.ydim; i++){
            this.mat[i] = new Array(this.xdim);
        }
    }
    get x(){
        return this.xdim;
    }
    get y(){
        return this.ydim;
    }
}

module.exports = Matrix;