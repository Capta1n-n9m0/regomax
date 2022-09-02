


class Matrix{
    constructor(x, y, diag = 0) {
        let i, j;
        this.xdim = x;
        this.ydim = y;
        this.init_mem();
        for(i = 0; i < this.ydim; i++){
            for(j = 0; j < this.xdim; j++){
                this.mat[i].push(0);
            }
        }
        if(diag !== 0){
            const n = x<y ? x : y;
            for(i = 0; i < n; i++){
                this.mat[i][i] = diag;
            }
        }
    }
    init_mem(){
        this.mat = [];
        for(let i = 0; i < this.ydim; i++){
            this.mat.push([]);
        }
    }
}

module.exports = Matrix;